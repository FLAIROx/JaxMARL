from smax.environments.mini_smac.mini_smac_env import MiniSMAC, State
from smax.environments.mini_smac.heuristic_enemy import create_heuristic_policy
from smax.environments.multi_agent_env import MultiAgentEnv
import chex
from typing import Dict, Optional, Tuple
import jax.numpy as jnp
import jax
from functools import partial


class EnemyMiniSMAC(MultiAgentEnv):
    """Class that presents the MiniSMAC environment as a single-player
    (but still multi-agent) environment. Functions like a wrapper, but
    not linked with any of the wrapper code because that is used differently."""

    def __init__(self, **env_kwargs):
        self._env = MiniSMAC(**env_kwargs)
        # only one team
        self.num_agents = self._env.num_agents_per_team
        self.num_agents_per_team = self._env.num_agents_per_team
        self.agents = [f"ally_{i}" for i in range(self.num_agents)]
        self.enemy_agents = [f"enemy_{i}" for i in range(self.num_agents)]
        self.all_agents = self.agents + self.enemy_agents
        self.observation_spaces = {
            i: self._env.observation_spaces[i] for i in self.agents
        }
        self.action_spaces = {i: self._env.action_spaces[i] for i in self.agents}

    def __getattr__(self, name: str):
        return getattr(self._env, name)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], State]:
        obs, state = self._env.reset(key)
        obs = {agent: obs[agent] for agent in self.agents}
        return obs, state

    def get_enemy_actions(self, key, enemy_obs):
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def step_env(self, key: chex.PRNGKey, state: State, actions: Dict[str, chex.Array]):
        obs = self._env.get_obs(state)
        enemy_obs = jnp.array([obs[agent] for agent in self.enemy_agents])
        key, action_key = jax.random.split(key)
        enemy_actions = self.get_enemy_actions(action_key, enemy_obs)

        actions = {k: v.squeeze() for k, v in actions.items()}
        actions = {**enemy_actions, **actions}
        obs, state, rewards, dones, infos = self._env.step_env(key, state, actions)
        obs = {agent: obs[agent] for agent in self.agents}
        rewards = {agent: rewards[agent] for agent in self.agents}
        all_done = dones["__all__"]
        dones = {agent: dones[agent] for agent in self.agents}
        dones["__all__"] = all_done

        return obs, state, rewards, dones, infos

    def get_all_unit_obs(self, state: State):
        return self._env.get_obs(state)

    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        obs = self.get_all_unit_obs(state)
        return {agent: obs[agent] for agent in self.agents}


class HeuristicEnemyMiniSMAC(EnemyMiniSMAC):
    def __init__(self, enemy_shoots=True, **env_kwargs):
        super().__init__(**env_kwargs)
        self.enemy_shoots = enemy_shoots

    def get_enemy_actions(self, key, enemy_obs):
        heuristic_policy = create_heuristic_policy(
            self._env, 1, shoot=self.enemy_shoots
        )
        heuristic_action_key = jax.random.split(key, num=self.num_agents_per_team)
        enemy_actions = jax.vmap(heuristic_policy)(heuristic_action_key, enemy_obs)
        enemy_actions = {
            agent: enemy_actions[self._env.agent_ids[agent] % self.num_agents_per_team]
            for agent in self.enemy_agents
        }
        return enemy_actions


class LearnedPolicyEnemyMiniSMAC(EnemyMiniSMAC):
    def __init__(self, policy, params, **env_kwargs):
        super().__init__(**env_kwargs)
        self.policy = policy
        self.params = params

    def get_enemy_actions(self, key, enemy_obs):
        pi, _ = self.policy.apply(self.params, enemy_obs)
        enemy_actions = pi.sample(seed=key)
        enemy_actions = {
            agent: enemy_actions[self._env.agent_ids[agent] % self.num_agents_per_team]
            for agent in self.enemy_agents
        }
        enemy_actions = {k: v.squeeze() for k, v in enemy_actions.items()}
        return enemy_actions
