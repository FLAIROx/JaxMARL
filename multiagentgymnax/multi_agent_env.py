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
    pass 

@struct.dataclass
class EnvParams:
    pass

class MultiAgentEnv(object):
    
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
    ):
        if params is None:
            params = self.default_params
            
        return self.reset_env(key, params)
    
    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, 
        key: chex.PRNGKey, 
        states: State, 
        actions: chex.Array, 
        params: EnvParams
    ):
        key, key_reset = jax.random.split(key)
        obs_st, states_st, rewards, infos = self.step_env(
            key, states, actions, params
        )
        #out = jax.lax.cond(states_st.ep_done, self.reset_env, lambda x: x,  ) TODO
        
        #jax.debug.print('ep done {d} {s} ', d=states_st.ep_done, s=states_st)
        obs_re, states_re = self.reset_env(states.pos.shape[0], key_reset, params)  
        # Auto-reset environment based on termination
        states = jax.tree_map(
            lambda x, y: jax.lax.select(states_st.ep_done, x, y), states_re, states_st
        )
        obs = jax.lax.select(states_re.ep_done, obs_re, obs_st) # BUG fix this, need to use tree map =-- or do we..?
        return obs, states, rewards, states_st.done, infos
    
    def reset_env(
        self, key, params
    ):
        raise NotImplementedError
    
    def step_env(
        self, key, states, actions, params
    ):
        raise NotImplementedError
    
    def observation_space(self, agent):
        return self.observation_spaces[agent]
    
    def action_space(self, agent):
        return self.action_spaces[agent]