from smax.environments.mini_smac.mini_smac_env import MiniSMAC, State, EnvParams
from smax.environments.mini_smac.heuristic_enemy import create_heuristic_policy
from smax.environments.multi_agent_env import MultiAgentEnv
import chex
from typing import Dict, Optional, Tuple
import jax.numpy as jnp
import jax
from functools import partial


class HeuristicEnemyMiniSMAC(MultiAgentEnv):
    """Class that presents the MiniSMAC environment as a single-player
    (but still multi-agent) environment. Functions like a wrapper, but
    not linked with any of the wrapper code because that is used differently."""

    def __init__(self, num_agents_per_team=5, world_steps_per_env_step=8):
        self._env = MiniSMAC(
            num_agents_per_team=num_agents_per_team,
            world_steps_per_env_step=world_steps_per_env_step,
        )
        # only one team
        self.num_agents = num_agents_per_team
        self.num_agents_per_team = num_agents_per_team
        self.agents = [f"ally_{i}" for i in range(self.num_agents)]
        self.enemy_agents = [f"enemy_{i}" for i in range(self.num_agents)]
        self.all_agents = self.agents + self.enemy_agents
        self.observation_spaces = {
            i: self._env.observation_spaces[i] for i in self.agents
        }
        self.action_spaces = {i: self._env.action_spaces[i] for i in self.agents}

    def __getattr__(self, name: str):
        return getattr(self._env, name)

    @property
    def default_params(self):
        return EnvParams(
            num_agents_per_team=5,
            map_width=32,
            map_height=32,
            world_steps_per_env_step=8,
            unit_velocity=3.15,
            unit_type_attacks=jnp.array([0.013]),
            time_per_step=1.0 / 16,
            won_battle_bonus=1,
            unit_type_attack_ranges=jnp.array([5.0]),
            unit_type_sight_ranges=jnp.array([9.0]),
            max_steps=100,
        )

    @partial(jax.jit, static_argnums=(0,))
    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[Dict[str, chex.Array], State]:
        obs, state = self._env.reset_env(key, params)
        obs = {agent: obs[agent] for agent in self.agents}
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self,
        key: chex.PRNGKey,
        state: State,
        actions: Dict[str, chex.Array],
        params: EnvParams,
    ):
        heuristic_policy = create_heuristic_policy(self._env, params, 1)
        obs = self._env.get_obs(state, params)
        enemy_obs = jnp.array([obs[agent] for agent in self.enemy_agents])
        key, heuristic_action_key = jax.random.split(key)
        heuristic_action_key = jax.random.split(
            heuristic_action_key, num=self.num_agents_per_team
        )
        enemy_actions = jax.vmap(heuristic_policy)(heuristic_action_key, enemy_obs)
        enemy_actions = {
            agent: enemy_actions[self._env.agent_ids[agent] % self.num_agents_per_team]
            for agent in self.enemy_agents
        }
        actions = {k: v.squeeze() for k, v in actions.items()}
        actions = {**enemy_actions, **actions}
        obs, state, rewards, dones, infos = self._env.step_env(
            key, state, actions, params
        )
        obs = {agent: obs[agent] for agent in self.agents}
        rewards = {agent: rewards[agent] for agent in self.agents}
        all_done = dones["__all__"]
        dones = {agent: dones[agent] for agent in self.agents}
        dones["__all__"] = all_done

        return obs, state, rewards, dones, infos

    def get_all_unit_obs(self, state: State, params: EnvParams):
        return self._env.get_obs(state, params)

    def get_obs(self, state: State, params: EnvParams) -> Dict[str, chex.Array]:
        obs = self.get_all_unit_obs(state, params)
        return {agent: obs[agent] for agent in self.agents}
