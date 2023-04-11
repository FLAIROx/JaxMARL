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
     


class MultiAgentEnv(object):  # NOTE use abc base calss
    
    def __init__(self,
                 num_agents: int,
    ) -> None:
        self.num_agents = num_agents
        self.observation_spaces = dict()
        self.action_spaces = dict() 
        
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey):
            
        return self.reset_env(key)
    
    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, 
        key: chex.PRNGKey, 
        state: State, 
        actions: chex.Array, 
    ):
        key, key_reset = jax.random.split(key)
        obs_st, states_st, rewards, infos = self.step_env(
            key, state, actions
        )
        #out = jax.lax.cond(states_st.ep_done, self.reset_env, lambda x: x,  ) TODO
        
        #jax.debug.print('ep done {d} {s} ', d=states_st.ep_done, s=states_st)
        print('reset env', self.reset_env(key_reset) )
        obs_re, states_re = self.reset_env(key_reset)  
        # Auto-reset environment based on termination
        print('states', states_st, '\n', states_re)
        state = jax.tree_map(
            lambda x, y: jax.lax.select(jnp.all(states_st.done), x, y), states_re, states_st
        )
        obs = jax.lax.select(jnp.all(states_st.done), obs_re, obs_st) # BUG fix this, need to use tree map =-- or do we..?
        return obs, state, rewards, states_st.done, infos
    
    def reset_env(
        self, key
    ) -> Tuple[chex.Array, State]:
        raise NotImplementedError
    
    def step_env(
        self, key: chex.PRNGKey, state: State, actions: chex.Array
    ):
        raise NotImplementedError
    
    def observation_space(self, agent):
        return self.observation_spaces[agent]
    
    def action_space(self, agent):
        return self.action_spaces[agent]
    
    # == PLOTTING ==
    def render(self, state):
        raise NotImplementedError
    
    def close(self, state):
        raise NotImplementedError
