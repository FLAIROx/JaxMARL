""" 
Abstract base class for multi agent gym environments with JAX
"""

import jax 
import jax.numpy as jnp
import chex 
from functools import partial
from flax import struct
from typing import Tuple, Optional

@struct.dataclass
class State:
    done: chex.Array
    step: int
     
@struct.dataclass
class EnvParams:
    max_steps: int


class MultiAgentEnv(object):  # NOTE use abc base calss
    
    def __init__(self,
                 num_agents: int,
    ) -> None:
        self.num_agents = num_agents
        self.observation_spaces = dict()
        self.action_spaces = dict() 

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()
        
    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[EnvParams] = None
    ) -> Tuple[chex.Array, State]:
        if params is None:
            params = self.default_params

        return self.reset_env(key, params)
    
    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, 
        key: chex.PRNGKey, 
        state: State, 
        actions: chex.Array, 
        params: Optional[EnvParams] = None,
    ) -> Tuple[chex.Array, State, chex.Array, chex.Array, dict]:
        if params is None:
            params = self.default_params
        key, key_reset = jax.random.split(key)
        obs_st, states_st, rewards, infos = self.step_env(
            key, state, actions, params
        )
        
        obs_re, states_re = self.reset_env(key_reset, params)  
        
        # Auto-reset environment based on termination
        state = jax.tree_map(
            lambda x, y: jax.lax.select(jnp.all(states_st.done), x, y), states_re, states_st
        )
        #obs = jax.lax.select(jnp.all(states_st.done), obs_re, obs_st) # BUG fix this, need to use tree map =-- or do we..?
        print('obs', obs_st, obs_re)
        obs = jax.tree_map(
            lambda x, y: jax.lax.select(jnp.all(states_st.done), x, y), obs_re, obs_st
        )
        return obs, state, rewards, states_st.done, infos
    
    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, State]:
        raise NotImplementedError
    
    def step_env(
        self, key: chex.PRNGKey, state: State, actions: chex.Array, Params: EnvParams
    ) -> Tuple[chex.Array, State, chex.Array, dict]:
        raise NotImplementedError
    
    def observation_space(self, agent: str):
        return self.observation_spaces[agent]
    
    def action_space(self, agent: str):
        return self.action_spaces[agent]
    
    # == PLOTTING ==
    def render(self, state: State, params: EnvParams):
        raise NotImplementedError
    
    def close(self, state: State, params: EnvParams):
        raise NotImplementedError
