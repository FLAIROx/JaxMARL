import jax 
import jax.numpy as jnp
import chex
from typing import Tuple, Dict
from functools import partial
from multiagentgymnax.environments.mpe._mpe_utils.mpe_base_env import MPEBaseEnv, State, EnvParams
from multiagentgymnax.environments.mpe._mpe_utils.default_params import *
from gymnax.environments.spaces import Box

class SimpleMPE(MPEBaseEnv):
    
    def __init__(self,
                 num_agents=1,
                 num_landmarks=1,):
        
        dim_c = 0 # NOTE follows code rather than docs
        
        # Action and observation spaces
        agents = ["agent_{}".format(i) for i in range(num_agents)]
        landmarks = ["landmark {}".format(i) for i in range(num_landmarks)]
        
        action_spaces = {i: Box(0.0, 1.0, (5,)) for i in agents}
        observation_spaces = {i: Box(-jnp.inf, jnp.inf, (4,)) for i in agents }
        
        colour = [AGENT_COLOUR] * num_agents + [OBS_COLOUR] * num_landmarks
                
        super().__init__(num_agents=num_agents,
                        agents=agents,
                        num_landmarks=num_landmarks,
                        landmarks=landmarks,
                        action_spaces=action_spaces,
                        observation_spaces=observation_spaces,
                        dim_c=dim_c,
                        colour=colour)
                        
    @property
    def default_params(self) -> EnvParams:
        params = EnvParams(
            max_steps=MAX_STEPS,
            rad=jnp.concatenate([jnp.full((self.num_agents), 0.15),
                            jnp.full((self.num_landmarks), 0.2)]), # landmarks size?
            moveable=jnp.concatenate([jnp.full((self.num_agents), True), jnp.full((self.num_landmarks), False)]),
            silent = jnp.full((self.num_agents), 1),
            collide = jnp.full((self.num_entities), False),
            mass=jnp.full((self.num_entities), 1),
            accel = jnp.full((self.num_agents), AGENT_ACCEL),
            max_speed = jnp.concatenate([jnp.full((self.num_agents), AGENT_MAX_SPEED),
                                jnp.full((self.num_landmarks), 0.0)]),
            u_noise=jnp.full((self.num_agents), 0),
            c_noise=jnp.full((self.num_agents), 0),
            damping=DAMPING,  # physical damping
            contact_force=CONTACT_FORCE,  # contact response parameters
            contact_margin=CONTACT_MARGIN,
            dt=DT,            
        )
        return params
