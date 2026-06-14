from functools import partial
from typing import Tuple, Union

import chex
import jax
from flax import struct
from jaxtyping import PRNGKeyArray


@struct.dataclass
class EnvState:
    time: int


@struct.dataclass
class EnvParams:
    max_episode_steps: int


class Environment(object):
    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    @partial(jax.jit, static_argnums=(0, 4))
    def step(
        self, key: PRNGKeyArray, state: EnvState, action: Union[int, float]
    ) -> Tuple[chex.ArrayTree, EnvState, float, bool]:
        """Performs step transitions in the environment."""
        # Use default env parameters if no others specified
        if hasattr(self, "params"):
            params = self.params
        else:
            params = self.default_params

        key, key_reset = jax.random.split(key)
        obs_st, state_st, reward, done, info = self.step_env(key, state, action)

        if hasattr(params, "singleton_seed") and params.singleton_seed >= 0:
            key_reset = jax.random.PRNGKey(params.singleton_seed)

        obs_re, state_re = self.reset_env(key_reset)

        # Auto-reset environment based on termination
        state = jax.tree.map(
            lambda x, y: jax.lax.select(done, x, y), state_re, state_st
        )
        obs = jax.tree.map(lambda x, y: jax.lax.select(done, x, y), obs_re, obs_st)

        return obs, state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self,
        key: PRNGKeyArray,
    ) -> Tuple[chex.ArrayTree, EnvState]:
        """Performs resetting of environment."""
        # Use default env parameters if no others specified
        if hasattr(self, "params"):
            params = self.params
        else:
            params = self.default_params

        if hasattr(params, "singleton_seed") and params.singleton_seed >= 0:
            key = jax.random.PRNGKey(params.singleton_seed)
        obs, state = self.reset_env(key)
        return obs, state

    def step_env(
        self,
        key: PRNGKeyArray,
        state: EnvState,
        action: Union[int, float],
    ) -> Tuple[chex.ArrayTree, EnvState, float, bool, dict]:
        """Environment-specific step transition."""
        raise NotImplementedError

    def reset_env(self, key: PRNGKeyArray) -> Tuple[chex.ArrayTree, EnvState]:
        """Environment-specific reset."""
        raise NotImplementedError

    def get_obs(self, state: EnvState) -> chex.ArrayTree:
        """Applies observation function to state."""
        raise NotImplementedError

    def is_terminal(self, state: EnvState) -> bool:
        """Check whether state is terminal."""
        raise NotImplementedError

    @property
    def name(self) -> str:
        """Environment name."""
        return type(self).__name__

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        raise NotImplementedError

    def action_space(self):
        """Action space of the environment."""
        raise NotImplementedError

    def observation_space(self):
        """Observation space of the environment."""
        raise NotImplementedError

    def state_space(self):
        """State space of the environment."""
        raise NotImplementedError

    def max_episode_steps(self):
        """Maximum number of time steps in environment."""
        raise NotImplementedError
