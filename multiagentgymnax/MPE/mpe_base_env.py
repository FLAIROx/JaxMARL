""" 
Base env class for MPE PettingZoo envs.
"""

import jax
import jax.numpy as jnp 


"""
has both a physical and communication action

"""


class SimpleMPEEnv(MultiAgentEnv):
    
    
    def __init__(self, 
                 num_agents: int,
                 ) -> None:
        super().__init__(num_agents)
        
        
        self.communication = False 
        self.competitive = False

    def step_env(self, key, states, actions, params):
        
    
    
    def _was_dead_step(self, action: None):
        pass
    
    