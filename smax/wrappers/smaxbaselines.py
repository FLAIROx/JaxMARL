""" Wrappers for use with SMAX baselines. """
import jax
import jax.numpy as jnp
import chex
import numpy as np
from flax import struct
from functools import partial
#from gymnax.environments import environment, spaces
from typing import Optional, Tuple, Union
from smax.environments import multi_agent_env

class SMAXWrapper(object):
    """ Base class for all SMAX wrappers. """
    
    def __init__(self, env: multi_agent_env.MultiAgentEnv):
        self._env = env 
        
    def __getattr__(self, name: str):
        return getattr(self._env, name)
    
    def _batchify(self, x: dict):
        x = jnp.stack([x[a] for a in self._env.agents])
        return x.reshape((self._env.num_agents, -1))
    
    def _batchify_floats(self, x: dict):
        return jnp.stack([x[a] for a in self._env.agents])
    

class HomogenousBatch(SMAXWrapper):
    """ Convert outputs from dicts to arrays, indexed by agent.
    Only works for domains with homogenous agents.
    NOTE not complete
    """
    def __init__(self, env):
        super().__init__(env)
        self.num_agents = self._env.num_agents
    
    def reset(
        self, key: chex.PRNGKey, params: Optional[multi_agent_env.EnvParams] = None
    ) -> Tuple[chex.Array, multi_agent_env.State]:
        obs, state = self._env.reset(key, params)
        obs = jnp.stack([obs[a] for a in self._env.agent_list])
        obs = jnp.reshape(obs, (self._env.num_agents, -1))
        return obs, state
        
    def step(
        self,
        key,
        state,
        actions,
        params
    ):
        actions = {a: actions[i] for i, a in enumerate(self._env.agents)}
        obs, state, reward, done, info = self._env.step(key, state, actions, params)
        obs = jnp.reshape(obs, (self._env.num_agents, -1))
        

@struct.dataclass
class LogEnvState:
    env_state: multi_agent_env.State
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int


class LogWrapper(SMAXWrapper):
    """Log the episode returns and lengths.
    NOTE for now for envs where agents terminate at the same time.
    """

    def __init__(self, env: multi_agent_env.MultiAgentEnv):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[multi_agent_env.EnvParams] = None
    ) -> Tuple[chex.Array, multi_agent_env.State]:
        obs, env_state = self._env.reset(key, params)
        state = LogEnvState(env_state, 
                            jnp.zeros((self._env.num_agents,)),
                            jnp.zeros((self._env.num_agents,)),
                            jnp.zeros((self._env.num_agents,)),
                            jnp.zeros((self._env.num_agents,)))
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: LogEnvState,
        action: Union[int, float],
        params: Optional[multi_agent_env.EnvParams] = None,
    ) -> Tuple[chex.Array, LogEnvState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        ep_done = done["__all__"]
        new_episode_return = state.episode_returns + self._batchify_floats(reward)
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
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["returned_episode"] = jnp.full((self._env.num_agents,), ep_done)
        return obs, state, reward, done, info
