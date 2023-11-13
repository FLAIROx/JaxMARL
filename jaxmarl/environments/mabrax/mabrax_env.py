from typing import Dict, Literal, Optional, Tuple
import chex
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from gymnax.environments import spaces
from brax import envs
import jax
import jax.numpy as jnp
from functools import partial

from .mappings import _agent_action_mapping, _agent_observation_mapping

# TODO: move homogenisation to a separate wrapper


class MABraxEnv(MultiAgentEnv):
    def __init__(
        self,
        env_name: str,
        episode_length: int = 1000,
        action_repeat: int = 1,
        auto_reset: bool = True,
        homogenisation_method: Optional[Literal["max", "concat"]] = None,
        backend: str = "positional",
        **kwargs
    ):
        """Multi-Agent Brax environment.

        Args:
            env_name: Name of the environment to be used.
            episode_length: Length of an episode. Defaults to 1000.
            action_repeat: How many repeated actions to take per environment
                step. Defaults to 1.
            auto_reset: Whether to automatically reset the environment when
                an episode ends. Defaults to True.
            homogenisation_method: Method to homogenise observations and actions
                across agents. If None, no homogenisation is performed, and
                observations and actions are returned as is. If "max", observations
                and actions are homogenised by taking the maximum dimension across
                all agents and zero-padding the rest. In this case, the index of the
                agent is prepended to the observation as a one-hot vector. If "concat",
                observations and actions are homogenised by masking the dimensions of
                the other agents with zeros in the full observation and action vectors.
                Defaults to None.
        """
        base_env_name = env_name.split("_")[0]
        env = envs.create(
            base_env_name, episode_length, action_repeat, auto_reset, backend=backend, **kwargs
        )
        self.env = env
        self.episode_length = episode_length
        self.action_repeat = action_repeat
        self.auto_reset = auto_reset
        self.homogenisation_method = homogenisation_method
        self.agent_obs_mapping = _agent_observation_mapping[env_name]
        self.agent_action_mapping = _agent_action_mapping[env_name]
        self.agents = list(self.agent_obs_mapping.keys())

        self.num_agents = len(self.agent_obs_mapping)
        obs_sizes = {
            agent: self.num_agents
            + max([o.size for o in self.agent_obs_mapping.values()])
            if homogenisation_method == "max"
            else self.env.observation_size
            if homogenisation_method == "concat"
            else obs.size
            for agent, obs in self.agent_obs_mapping.items()
        }
        act_sizes = {
            agent: max([a.size for a in self.agent_action_mapping.values()])
            if homogenisation_method == "max"
            else self.env.action_size
            if homogenisation_method == "concat"
            else act.size
            for agent, act in self.agent_action_mapping.items()
        }

        self.observation_spaces = {
            agent: spaces.Box(
                -jnp.inf,
                jnp.inf,
                shape=(obs_sizes[agent],),
            )
            for agent in self.agents
        }
        self.action_spaces = {
            agent: spaces.Box(
                -1.0,
                1.0,
                shape=(act_sizes[agent],),
            )
            for agent in self.agents
        }

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], envs.State]:
        state = self.env.reset(key)
        return self.get_obs(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self,
        key: chex.PRNGKey,
        state: envs.State,
        actions: Dict[str, chex.Array],
    ) -> Tuple[
        Dict[str, chex.Array], envs.State, Dict[str, float], Dict[str, bool], Dict
    ]:
        global_action = self.map_agents_to_global_action(actions)
        next_state = self.env.step(state, global_action)  # type: ignore
        observations = self.get_obs(next_state)
        rewards = {agent: next_state.reward for agent in self.agents}
        rewards["__all__"] = next_state.reward
        dones = {agent: next_state.done.astype(jnp.bool_) for agent in self.agents}
        dones["__all__"] = next_state.done.astype(jnp.bool_)
        return (
            observations,
            next_state,  # type: ignore
            rewards,
            dones,
            next_state.info,
        )

    def get_obs(self, state: envs.State) -> Dict[str, chex.Array]:
        """Extracts agent observations from the global state.

        Args:
            state: Global state of the environment.

        Returns:
            A dictionary of observations for each agent.
        """
        return self.map_global_obs_to_agents(state.obs)

    def map_agents_to_global_action(
        self, agent_actions: Dict[str, jnp.ndarray]
    ) -> jnp.ndarray:
        global_action = jnp.zeros(self.env.action_size)
        for agent_name, action_indices in self.agent_action_mapping.items():
            if self.homogenisation_method == "max":
                global_action = global_action.at[action_indices].set(
                    agent_actions[agent_name][: action_indices.size]
                )
            elif self.homogenisation_method == "concat":
                global_action = global_action.at[action_indices].set(
                    agent_actions[agent_name][action_indices]
                )
            else:
                global_action = global_action.at[action_indices].set(
                    agent_actions[agent_name]
                )
        return global_action

    def map_global_obs_to_agents(
        self, global_obs: jnp.ndarray
    ) -> Dict[str, jnp.ndarray]:
        """Maps the global observation vector to the individual agent observations.

        Args:
            global_obs: The global observation vector.

        Returns:
            A dictionary mapping agent names to their observations. The mapping method
            is determined by the homogenisation_method parameter.
        """
        agent_obs = {}
        for agent_idx, (agent_name, obs_indices) in enumerate(
            self.agent_obs_mapping.items()
        ):
            if self.homogenisation_method == "max":
                # Vector with the agent idx one-hot encoded as the first num_agents
                # elements and then the agent's own observations (zero padded to
                # the size of the largest agent observation vector)
                agent_obs[agent_name] = (
                    jnp.zeros(
                        self.num_agents
                        + max([v.size for v in self.agent_obs_mapping.values()])
                    )
                    .at[agent_idx]
                    .set(1)
                    .at[agent_idx + 1 : agent_idx + 1 + obs_indices.size]
                    .set(global_obs[obs_indices])
                )
            elif self.homogenisation_method == "concat":
                # Zero vector except for the agent's own observations
                # (size of the global observation vector)
                agent_obs[agent_name] = (
                    jnp.zeros(global_obs.shape)
                    .at[obs_indices]
                    .set(global_obs[obs_indices])
                )
            else:
                # Just agent's own observations
                agent_obs[agent_name] = global_obs[obs_indices]
        return agent_obs

    @property
    def sys(self):
        return self.env.sys


class Ant(MABraxEnv):
    def __init__(self, **kwargs):
        super().__init__("ant_4x2", **kwargs)


class HalfCheetah(MABraxEnv):
    def __init__(self, **kwargs):
        super().__init__("halfcheetah_6x1", **kwargs)


class Hopper(MABraxEnv):
    def __init__(self, **kwargs):
        super().__init__("hopper_3x1", **kwargs)


class Humanoid(MABraxEnv):
    def __init__(self, **kwargs):
        super().__init__("humanoid_9|8", **kwargs)


class Walker2d(MABraxEnv):
    def __init__(self, **kwargs):
        super().__init__("walker2d_2x3", **kwargs)
