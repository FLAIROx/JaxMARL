import dataclasses
from jaxmarl.environments.smax.smax_env import SMAX
from jaxmarl.environments.smax.smax_env import State as SMAXState
from jaxmarl.environments.smax.heuristic_enemy import (
    create_heuristic_policy,
    get_heuristic_policy_initial_state,
)
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
import chex
from typing import Dict, Optional, Tuple
import jax.numpy as jnp
import jax
from flax.struct import dataclass
from functools import partial


@dataclass
class State:
    # underlying jaxmarl env state
    state: ...
    # the enemy policy state. Needed for recurrent policies or
    # remembering details about previous observations for heuristics.
    enemy_policy_state: ...


class EnemySMAX(MultiAgentEnv):
    """Class that presents the SMAX environment as a single-player
    (but still multi-agent) environment. Functions like a wrapper, but
    not linked with any of the wrapper code because that is used differently."""

    def __init__(self, **env_kwargs):
        self._env = SMAX(**env_kwargs)
        # only one team
        self.num_agents = self._env.num_allies
        self.num_enemies = self._env.num_enemies
        # want to provide a consistent API between this and SMAX
        self.num_allies = self._env.num_allies
        self.agents = [f"ally_{i}" for i in range(self.num_agents)]
        self.enemy_agents = [f"enemy_{i}" for i in range(self.num_enemies)]
        self.all_agents = self.agents + self.enemy_agents
        self.observation_spaces = {
            i: self._env.observation_spaces[i] for i in self.agents
        }
        self.action_spaces = {i: self._env.action_spaces[i] for i in self.agents}

    def __getattr__(self, name: str):
        return getattr(self._env, name)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], State]:
        key, reset_key = jax.random.split(key)
        obs, state = self._env.reset(reset_key)
        enemy_policy_state = self.get_enemy_policy_initial_state(key)
        new_obs = {agent: obs[agent] for agent in self.agents}
        new_obs["world_state"] = obs["world_state"]
        return new_obs, State(state=state, enemy_policy_state=enemy_policy_state)

    def get_enemy_actions(self, key, enemy_policy_state, enemy_obs):
        raise NotImplementedError

    def get_enemy_policy_initial_state(self, key):
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0, 4))
    def step_env(
        self,
        key: chex.PRNGKey,
        state: State,
        actions: Dict[str, chex.Array],
        get_state_sequence=False,
    ):
        jaxmarl_state = state.state
        obs = self._env.get_obs(jaxmarl_state)
        enemy_obs = self._env.get_obs_unit_list(jaxmarl_state)
        enemy_obs = jnp.array([enemy_obs[agent] for agent in self.enemy_agents])
        key, action_key = jax.random.split(key)
        enemy_actions, enemy_policy_state = self.get_enemy_actions(
            action_key, state.enemy_policy_state, enemy_obs
        )
        enemy_actions = jnp.array([enemy_actions[i] for i in self.enemy_agents])
        actions = jnp.array([actions[i] for i in self.agents])
        enemy_movement_actions, enemy_attack_actions = (
            self._env._decode_discrete_actions(enemy_actions)
        )
        if self._env.action_type == "continuous":
            cont_actions = jnp.zeros((len(self.all_agents), 4))
            cont_actions = cont_actions.at[: self.num_allies].set(actions)
            key, action_key = jax.random.split(key)
            ally_movement_actions, ally_attack_actions = (
                self._env._decode_continuous_actions(
                    action_key, jaxmarl_state, cont_actions
                )
            )
            ally_movement_actions = ally_movement_actions[: self.num_allies]
            ally_attack_actions = ally_attack_actions[: self.num_allies]
        else:
            ally_movement_actions, ally_attack_actions = (
                self._env._decode_discrete_actions(actions)
            )

        movement_actions = jnp.concatenate(
            [ally_movement_actions, enemy_movement_actions], axis=0
        )
        attack_actions = jnp.concatenate([ally_attack_actions, enemy_attack_actions], axis=0)

        if not get_state_sequence:
            obs, jaxmarl_state, rewards, dones, infos = self._env.step_env_no_decode(
                key,
                jaxmarl_state,
                (movement_actions, attack_actions),
                get_state_sequence=get_state_sequence,
            )
            new_obs = {agent: obs[agent] for agent in self.agents}
            new_obs["world_state"] = obs["world_state"]
            rewards = {agent: rewards[agent] for agent in self.agents}
            all_done = dones["__all__"]
            dones = {agent: dones[agent] for agent in self.agents}
            dones["__all__"] = all_done

            state = state.replace(
                enemy_policy_state=enemy_policy_state, state=jaxmarl_state
            )
            return new_obs, state, rewards, dones, infos
        else:
            states = self._env.step_env_no_decode(
                key,
                jaxmarl_state,
                (movement_actions, attack_actions),
                get_state_sequence=get_state_sequence,
            )
            return states

    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: State):
        avail_actions = self._env.get_avail_actions(state.state)
        return {agent: avail_actions[agent] for agent in self.agents}

    def get_all_unit_obs(self, state: State):
        return self._env.get_obs(state.state)

    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        obs = self.get_all_unit_obs(state)
        return {agent: obs[agent] for agent in self.agents}

    def get_world_state(self, state: State):
        return self._env.get_world_state(state.state)

    def is_terminal(self, state: State):
        return self._env.is_terminal(state.state)

    def expand_state_seq(self, state_seq):
        # TODO jit/scan this
        expanded_state_seq = []

        # TODO this actually can't take a key because recording this key is really hard
        # it's not exposed to the user so we can't ask them to store it. Not a problem
        # for now but will have to get creative in the future potentially.
        for key, state, actions in state_seq:
            # There is a split in the step function of MultiAgentEnv
            # We call split here so that the action key is the same.
            key, _ = jax.random.split(key)
            states = self.step_env(key, state, actions, get_state_sequence=True)
            states = list(map(SMAXState, *dataclasses.astuple(states)))
            viz_actions = {
                agent: states[-1].prev_attack_actions[i]
                for i, agent in enumerate(self.all_agents)
            }

            expanded_state_seq.append((key, state.state, viz_actions))
            expanded_state_seq.extend(
                zip([key] * len(states), states, [viz_actions] * len(states))
            )
            state = state.replace(
                state=state.state.replace(terminal=self.is_terminal(state))
            )
        return expanded_state_seq


class HeuristicEnemySMAX(EnemySMAX):
    def __init__(self, enemy_shoots=True, attack_mode="closest", **env_kwargs):
        super().__init__(**env_kwargs)
        self.enemy_shoots = enemy_shoots
        self.attack_mode = attack_mode
        self.heuristic_policy = create_heuristic_policy(
            self._env, 1, shoot=self.enemy_shoots, attack_mode=self.attack_mode
        )

    def get_enemy_policy_initial_state(self, key):
        return jax.tree_map(
            lambda *xs: jnp.stack(xs),
            *([get_heuristic_policy_initial_state()] * self.num_enemies),
        )

    def get_enemy_actions(self, key, policy_state, enemy_obs):
        heuristic_action_key = jax.random.split(key, num=self.num_enemies)
        enemy_actions, policy_state = jax.vmap(self.heuristic_policy)(
            heuristic_action_key, policy_state, enemy_obs
        )
        enemy_actions = {
            agent: enemy_actions[self._env.agent_ids[agent] - self.num_agents]
            for agent in self.enemy_agents
        }
        return enemy_actions, policy_state


class LearnedPolicyEnemySMAX(EnemySMAX):
    def __init__(self, policy, params, **env_kwargs):
        super().__init__(**env_kwargs)
        self.policy = policy
        self.params = params

    def get_enemy_policy_initial_state(self, key):
        return self.params

    def get_enemy_actions(self, key, policy_state, enemy_obs):
        pi, _ = self.policy.apply(policy_state, enemy_obs)
        enemy_actions = pi.sample(seed=key)
        enemy_actions = {
            agent: enemy_actions[self._env.agent_ids[agent] - self.num_agents]
            for agent in self.enemy_agents
        }
        enemy_actions = {k: v.squeeze() for k, v in enemy_actions.items()}
        return enemy_actions, policy_state
