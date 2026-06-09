"""Wrappers for use with jaxmarl baselines."""

import os
from functools import partial
from typing import Any, Dict, List, Mapping, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from flax.traverse_util import flatten_dict, unflatten_dict

# from gymnax.environments import environment, spaces
from gymnax.environments.spaces import Box as BoxGymnax
from gymnax.environments.spaces import Discrete as DiscreteGymnax
from jaxtyping import Array, Float, Int, Num, PRNGKeyArray
from safetensors.flax import load_file, save_file

from jaxmarl.environments.multi_agent_env import (
    Actions,
    Dones,
    Infos,
    MultiAgentEnv,
    Observations,
    Rewards,
    State,
)
from jaxmarl.environments.overcooked_v2 import OvercookedV2
from jaxmarl.environments.overcooked_v2.common import DynamicObject
from jaxmarl.environments.spaces import Box, Discrete, MultiDiscrete


def save_params(params: Dict, filename: Union[str, os.PathLike]) -> None:
    flattened_dict = flatten_dict(params, sep=",")
    save_file(flattened_dict, filename)


def load_params(filename: Union[str, os.PathLike]) -> Dict:
    flattened_dict = load_file(filename)
    return unflatten_dict(flattened_dict, sep=",")


class JaxMARLWrapper(object):
    """Base class for all jaxmarl wrappers."""

    def __init__(self, env: MultiAgentEnv):
        self._env = env

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def stack_agent_values(
        self, x: Mapping[str, Num[Array, "..."]]
    ) -> Num[Array, "..."]:
        return jnp.stack([x[a] for a in self._env.agents])


@struct.dataclass
class LogEnvState:
    env_state: State
    episode_returns: Num[Array, "..."]
    episode_lengths: Int[Array, "..."]
    returned_episode_returns: Num[Array, "..."]
    returned_episode_lengths: Int[Array, "..."]


class LogWrapper(JaxMARLWrapper):
    """Log the episode returns and lengths.
    NOTE for now for envs where agents terminate at the same time.
    """

    def __init__(self, env: MultiAgentEnv, replace_info: bool = False):
        super().__init__(env)
        self.replace_info = replace_info

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: PRNGKeyArray) -> Tuple[Observations, LogEnvState]:
        obs, env_state = self._env.reset(key)
        state = LogEnvState(
            env_state,
            jnp.zeros((self._env.num_agents,)),
            jnp.zeros((self._env.num_agents,)),
            jnp.zeros((self._env.num_agents,)),
            jnp.zeros((self._env.num_agents,)),
        )
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: PRNGKeyArray,
        state: LogEnvState,
        action: Actions,
    ) -> Tuple[Observations, LogEnvState, Rewards, Dones, Infos]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action
        )
        ep_done = done["__all__"]
        new_episode_return = state.episode_returns + self.stack_agent_values(reward)
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - ep_done),
            episode_lengths=new_episode_length * (1 - ep_done),
            returned_episode_returns=state.returned_episode_returns * (1 - ep_done)
            + new_episode_return * ep_done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - ep_done)
            + new_episode_length * ep_done,
        )
        if self.replace_info:
            info = {}
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["returned_episode"] = jnp.full((self._env.num_agents,), ep_done)
        return obs, state, reward, done, info


@struct.dataclass
class OvercookedV2LogEnvState(LogEnvState):
    returned_episode_recipe_returns: Dict[str, Float[Array, "..."]]


class OvercookedV2LogWrapper(JaxMARLWrapper):
    def __init__(self, env: OvercookedV2, replace_info: bool = False):
        super().__init__(env)
        self.replace_info = replace_info

        self.recipe_dict = {
            f"{recipe[0]}_{recipe[1]}_{recipe[2]}": DynamicObject.get_recipe_encoding(
                recipe
            )
            for recipe in env.possible_recipes
        }

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: PRNGKeyArray) -> Tuple[Observations, OvercookedV2LogEnvState]:
        obs, env_state = self._env.reset(key)

        recipe_returns = {
            r: jnp.zeros((self._env.num_agents,)) for r in self.recipe_dict
        }

        state = OvercookedV2LogEnvState(
            env_state,
            jnp.zeros((self._env.num_agents,)),
            jnp.zeros((self._env.num_agents,)),
            jnp.zeros((self._env.num_agents,)),
            jnp.zeros((self._env.num_agents,)),
            recipe_returns,
        )
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: PRNGKeyArray,
        state: OvercookedV2LogEnvState,
        action: Actions,
    ) -> Tuple[Observations, OvercookedV2LogEnvState, Rewards, Dones, Infos]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action
        )
        ep_done = done["__all__"]
        new_episode_return = state.episode_returns + self.stack_agent_values(reward)
        new_episode_length = state.episode_lengths + 1

        updated_recipe_returns = {
            id: jax.lax.select(
                (state.env_state.recipe == self.recipe_dict[id]) & ep_done,
                new_episode_return,
                old_episode_return,
            )
            for id, old_episode_return in state.returned_episode_recipe_returns.items()
        }

        state = OvercookedV2LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - ep_done),
            episode_lengths=new_episode_length * (1 - ep_done),
            returned_episode_returns=jax.lax.select(
                ep_done, new_episode_return, state.returned_episode_returns
            ),
            returned_episode_lengths=jax.lax.select(
                ep_done, new_episode_length, state.returned_episode_lengths
            ),
            returned_episode_recipe_returns=updated_recipe_returns,
        )
        if self.replace_info:
            info = {}
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["returned_episode"] = jnp.full((self._env.num_agents,), ep_done)
        info["returned_episode_recipe_returns"] = state.returned_episode_recipe_returns
        return obs, state, reward, done, info


class MPELogWrapper(LogWrapper):
    """Times reward signal by number of agents within the environment,
    to match the on-policy codebase."""

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: PRNGKeyArray,
        state: LogEnvState,
        action: Actions,
    ) -> Tuple[Observations, LogEnvState, Rewards, Dones, Infos]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action
        )
        rewardlog = jax.tree.map(
            lambda x: x * self._env.num_agents, reward
        )  # As per on-policy codebase
        ep_done = done["__all__"]
        new_episode_return = state.episode_returns + self.stack_agent_values(rewardlog)
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - ep_done),
            episode_lengths=new_episode_length * (1 - ep_done),
            returned_episode_returns=state.returned_episode_returns * (1 - ep_done)
            + new_episode_return * ep_done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - ep_done)
            + new_episode_length * ep_done,
        )
        if self.replace_info:
            info = {}
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["returned_episode"] = jnp.full((self._env.num_agents,), ep_done)
        return obs, state, reward, done, info


@struct.dataclass
class SMAXLogEnvState(LogEnvState):
    won_episode: int
    returned_won_episode: int


class SMAXLogWrapper(JaxMARLWrapper):
    def __init__(self, env: MultiAgentEnv, replace_info: bool = False):
        super().__init__(env)
        self.replace_info = replace_info

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: PRNGKeyArray) -> Tuple[Observations, SMAXLogEnvState]:
        obs, env_state = self._env.reset(key)
        state = SMAXLogEnvState(
            env_state,
            jnp.zeros((self._env.num_agents,)),
            jnp.zeros((self._env.num_agents,)),
            jnp.zeros((self._env.num_agents,)),
            jnp.zeros((self._env.num_agents,)),
            jnp.zeros((self._env.num_agents,)),
            jnp.zeros((self._env.num_agents,)),
        )
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: PRNGKeyArray,
        state: SMAXLogEnvState,
        action: Actions,
    ) -> Tuple[Observations, SMAXLogEnvState, Rewards, Dones, Infos]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action
        )
        ep_done = done["__all__"]
        batch_reward = self.stack_agent_values(reward)
        new_episode_return = state.episode_returns + self.stack_agent_values(reward)
        new_episode_length = state.episode_lengths + 1
        new_won_episode = (batch_reward >= 1.0).astype(jnp.float32)
        state = SMAXLogEnvState(
            env_state=env_state,
            won_episode=new_won_episode * (1 - ep_done),
            episode_returns=new_episode_return * (1 - ep_done),
            episode_lengths=new_episode_length * (1 - ep_done),
            returned_episode_returns=state.returned_episode_returns * (1 - ep_done)
            + new_episode_return * ep_done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - ep_done)
            + new_episode_length * ep_done,
            returned_won_episode=state.returned_won_episode * (1 - ep_done)
            + new_won_episode * ep_done,
        )
        if self.replace_info:
            info = {}
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["returned_won_episode"] = state.returned_won_episode
        info["returned_episode"] = jnp.full((self._env.num_agents,), ep_done)
        return obs, state, reward, done, info


def get_space_dim(space):
    # get the proper action/obs space from Discrete-MultiDiscrete-Box spaces
    if isinstance(space, (DiscreteGymnax, Discrete)):
        return space.n
    elif isinstance(space, (BoxGymnax, Box, MultiDiscrete)):
        return np.prod(space.shape)
    else:
        print(space)
        raise NotImplementedError(
            "Current wrapper works only with Discrete/MultiDiscrete/Box action and obs spaces"
        )


class CTRolloutManager(JaxMARLWrapper):
    """
    Rollout Manager for Centralized Training of with Parameters Sharing. Used by JaxMARL Q-Learning Baselines.
    - Batchify multiple environments (the number of parallel envs is defined by batch_size in __init__).
    - Adds a global state (obs["__all__"]) and a global reward (rewards["__all__"]) in the env.step returns.
    - Pads the observations of the agents in order to have all the same length.
    - Adds an agent id (one hot encoded) to the observation vectors.

    By default:
    - global_state is the concatenation of all agents' observations.
    - global_reward is the sum of all agents' rewards.
    """

    def __init__(
        self,
        env: MultiAgentEnv,
        batch_size: int,
        training_agents: List = None,
        preprocess_obs: bool = True,
    ):
        super().__init__(env)

        self.batch_size = batch_size

        # the agents to train could differ from the total trainable agents in the env (f.i. if using pretrained agents)
        # it's important to know it in order to compute properly the default global rewards and state
        self.training_agents = (
            self.agents if training_agents is None else training_agents
        )
        self.preprocess_obs = preprocess_obs

        # batched action sampling
        self.batch_samplers = {
            agent: jax.jit(jax.vmap(self.action_space(agent).sample, in_axes=0))
            for agent in self.agents
        }

        # assumes the observations are flattened vectors
        self.max_obs_length = max(
            list(map(lambda x: get_space_dim(x), self.observation_spaces.values()))
        )
        self.max_action_space = max(
            list(map(lambda x: get_space_dim(x), self.action_spaces.values()))
        )
        self.obs_size = self.max_obs_length
        if self.preprocess_obs:
            self.obs_size += len(self.agents)

        # agents ids
        self.agents_one_hot = {
            a: oh for a, oh in zip(self.agents, jnp.eye(len(self.agents)))
        }
        # valid actions
        self.valid_actions = {a: jnp.arange(u.n) for a, u in self.action_spaces.items()}
        self.valid_actions_oh = {
            a: jnp.concatenate((jnp.ones(u.n), jnp.zeros(self.max_action_space - u.n)))
            for a, u in self.action_spaces.items()
        }

        # custom global state and rewards for specific envs
        if "smax" in env.name.lower():
            self.global_state = lambda obs, state: obs["world_state"]
            self.global_reward = lambda rewards: rewards[self.training_agents[0]]
            self.get_valid_actions = lambda state: jax.vmap(env.get_avail_actions)(
                state
            )
        elif "overcooked" in env.name.lower():
            self.global_state = lambda obs, state: jnp.concatenate(
                [obs[agent].flatten() for agent in self.agents], axis=-1
            )
            self.global_reward = lambda rewards: rewards[self.training_agents[0]]
        elif "hanabi" in env.name.lower():
            self.global_reward = lambda rewards: rewards[self.training_agents[0]]
            self.get_valid_actions = lambda state: jax.vmap(env.get_legal_moves)(state)

    @partial(jax.jit, static_argnums=0)
    def batch_reset(self, key: PRNGKeyArray) -> Tuple[Observations, State]:
        keys = jax.random.split(key, self.batch_size)
        return jax.vmap(self.wrapped_reset, in_axes=0)(keys)

    @partial(jax.jit, static_argnums=0)
    def batch_step(self, key: PRNGKeyArray, states: State, actions: Actions):
        keys = jax.random.split(key, self.batch_size)
        return jax.vmap(self.wrapped_step, in_axes=(0, 0, 0))(keys, states, actions)

    @partial(jax.jit, static_argnums=0)
    def wrapped_reset(self, key: PRNGKeyArray) -> Tuple[Observations, State]:
        obs_, state = self._env.reset(key)
        if self.preprocess_obs:
            obs = jax.tree.map(
                self._preprocess_obs,
                {agent: obs_[agent] for agent in self.agents},
                self.agents_one_hot,
            )
        else:
            obs = obs_
        obs["__all__"] = self.global_state(obs_, state)
        return obs, state

    @partial(jax.jit, static_argnums=0)
    def wrapped_step(
        self, key: PRNGKeyArray, state: State, actions: Actions
    ) -> Tuple[Observations, State, Rewards, Dones, Infos]:
        obs_, state, reward, done, infos = self._env.step(key, state, actions)
        if self.preprocess_obs:
            obs = jax.tree.map(
                self._preprocess_obs,
                {agent: obs_[agent] for agent in self.agents},
                self.agents_one_hot,
            )
            obs = jax.tree.map(
                lambda d, o: jnp.where(d, 0.0, o),
                {agent: done[agent] for agent in self.agents},
                obs,
            )  # ensure that the obs are 0s for done agents
        else:
            obs = obs_
        obs["__all__"] = self.global_state(obs_, state)
        reward["__all__"] = self.global_reward(reward)
        return obs, state, reward, done, infos

    @partial(jax.jit, static_argnums=0)
    def global_state(self, obs: Observations, state: State) -> Num[Array, "..."]:
        return jnp.concatenate([obs[agent] for agent in self.agents], axis=-1)

    @partial(jax.jit, static_argnums=0)
    def global_reward(self, reward: Rewards) -> Num[Array, "..."]:
        return jnp.stack([reward[agent] for agent in self.training_agents]).sum(axis=0)

    def batch_sample(self, key: PRNGKeyArray, agent: str) -> Int[Array, "..."]:
        return self.batch_samplers[agent](
            jax.random.split(key, self.batch_size)
        ).astype(int)

    @partial(jax.jit, static_argnums=0)
    def get_valid_actions(self, state: State) -> Dict[str, Int[Array, "..."]]:
        # default is to return the same valid actions one hot encoded for each env
        return {
            agent: jnp.tile(actions, self.batch_size).reshape(self.batch_size, -1)
            for agent, actions in self.valid_actions_oh.items()
        }

    @partial(jax.jit, static_argnums=0)
    def _preprocess_obs(self, arr: Array, extra_features: Array) -> Array:
        # flatten
        arr = arr.flatten()
        # pad the observation vectors to the maximum length
        pad_width = [(0, 0)] * (arr.ndim - 1) + [
            (0, max(0, self.max_obs_length - arr.shape[-1]))
        ]
        arr = jnp.pad(arr, pad_width, mode="constant", constant_values=0)
        # concatenate the extra features
        arr = jnp.concatenate((arr, extra_features), axis=-1)
        return arr
