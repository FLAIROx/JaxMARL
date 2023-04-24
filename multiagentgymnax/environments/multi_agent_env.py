""" 
Abstract base class for multi agent gym environments with JAX
Based on the Gymnax and PettingZoo APIs

"""

import jax 
import jax.numpy as jnp
from typing import Dict
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


class MultiAgentEnv(object):  
    
    def __init__(self,
                 num_agents: int,  
    ) -> None:
        """
            num_agents (int): maximum number of agents within the environment
        """
        self.num_agents = num_agents
        self.observation_spaces = dict()
        self.action_spaces = dict() 

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()
        
    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[EnvParams] = None
    ) -> Tuple[Dict[str, chex.Array], State]:
        """Performs resetting of the environment."""
        if params is None:
            params = self.default_params

        return self.reset_env(key, params)
    
    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, 
        key: chex.PRNGKey, 
        state: State, 
        actions: Dict[str, chex.Array], 
        params: Optional[EnvParams] = None,
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """Performs step transitions in the environment."""
        if params is None:
            params = self.default_params
        key, key_reset = jax.random.split(key)
        obs_st, states_st, rewards, dones, infos = self.step_env(
            key, state, actions, params
        )
        
        obs_re, states_re = self.reset_env(key_reset, params)  
        
        # Auto-reset environment based on termination
        states = jax.tree_map(
            lambda x, y: jax.lax.select(dones["__all__"], x, y), states_re, states_st
        )
        obs = jax.tree_map(
            lambda x, y: jax.lax.select(dones["__all__"], x, y), obs_re, obs_st
        )
        return obs, states, rewards, dones, infos
    
    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[Dict[str, chex.Array], State]:
        raise NotImplementedError
    
    def step_env(
        self, key: chex.PRNGKey, state: State, actions: Dict[str, chex.Array], Params: EnvParams
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        raise NotImplementedError
    
    def observation_space(self, agent: str):
        return self.observation_spaces[agent]
    
    def action_space(self, agent: str):
        return self.action_spaces[agent]
    
    # == PLOTTING ==
    def enable_render(self, state: State, params: EnvParams) -> None:
        raise NotImplementedError

    def render(self, state: State, params: Optional[EnvParams] = None) -> None:
        raise NotImplementedError
    
    def close(self, state: State, params: EnvParams) -> None:
        raise NotImplementedError
