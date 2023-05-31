import jax 
import jax.numpy as jnp
import chex
from typing import Tuple, Dict
from flax import struct
from functools import partial
from smax.environments.mpe.simple import SimpleMPE, TargetState, EnvParams
from smax.environments.mpe.default_params import *
from gymnax.environments.spaces import Box

SPEAKER = "speaker_0"
LISTENER = "listener_0"
AGENT_NAMES = [SPEAKER, LISTENER]

COLOUR_1 = jnp.array([0.65, 0.15, 0.15])
COLOUR_2 = jnp.array([0.15, 0.65, 0.15])
COLOUR_3 = jnp.array([0.15, 0.15, 0.65])



class SimpleSpeakerListenerMPE(SimpleMPE):
    
    def __init__(
        self,
        num_agents=2,
        num_landmarks=3,
    ):
        assert num_agents==2, "SimpleSpeakerListnerMPE only supports 2 agents"
        assert num_landmarks==3, "SimpleSpeakerListnerMPE only supports 3 landmarks"
        
        dim_c = 3
        # collaborative bool .. ?
        
        agents = AGENT_NAMES
        
        landmarks = ["landmark {}".format(i) for i in range(num_landmarks)]
        
        # Action and observation spaces
        action_spaces = {
            SPEAKER: Box(0.0, 1.0, (3,)),
            LISTENER: Box(0.0, 1.0, (5,)),
        }

        observation_spaces = {
            SPEAKER: Box(-jnp.inf, jnp.inf, (3,)),
            LISTENER: Box(-jnp.inf, jnp.inf, (11,)),
        }

        colour = [ADVERSARY_COLOUR] + [AGENT_COLOUR] + list(jnp.concatenate([COLOUR_1, COLOUR_2, COLOUR_3]))
        
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
            max_steps=25,
            rad=jnp.concatenate([jnp.full((self.num_agents), ADVERSARY_RADIUS),
                               jnp.full((self.num_landmarks), 0.04)]),
            moveable=jnp.concatenate([jnp.array([False]), jnp.array([True]), jnp.full((self.num_landmarks), False)]),
            silent = jnp.array([0, 1]),
            collide = jnp.full((self.num_entities), False),
            mass=jnp.full((self.num_entities), MASS),
            accel = jnp.full((self.num_agents), ACCEL),
            max_speed = jnp.concatenate([jnp.full((self.num_agents), MAX_SPEED),
                                jnp.full((self.num_landmarks), 0.0)]),
            u_noise=jnp.full((self.num_agents), 0),
            c_noise=jnp.full((self.num_agents), 0),
            damping=DAMPING,  # physical damping
            contact_force=CONTACT_FORCE,  # contact response parameters
            contact_margin=CONTACT_MARGIN,
            dt=DT,
        )
        return params
    
    def reset_env(self, key: chex.PRNGKey, params: EnvParams) -> Tuple[chex.Array, TargetState]:
        
        key_a, key_l, key_g = jax.random.split(key, 3)        
        
        p_pos = jnp.concatenate([
            jax.random.uniform(key_a, (self.num_agents, 2), minval=-1, maxval=+1),
            jax.random.uniform(key_l, (self.num_landmarks, 2), minval=-0.9, maxval=+0.9)
        ])
        
        g_idx = jax.random.randint(key_g, (), minval=0, maxval=self.num_landmarks)
        
        state = TargetState(
            p_pos=p_pos,
            p_vel=jnp.zeros((self.num_entities, self.dim_p)),
            c=jnp.zeros((self.num_agents, self.dim_c)),
            done=jnp.full((self.num_agents), False),
            step=0,
            goal=g_idx,
        )
        
        return self.get_obs(state, params), state
    
    def set_actions(self, actions: Dict, params: EnvParams):
        """ Extract u and c actions for all agents from actions Dict."""
        # NOTE only continuous actions
        u = jnp.zeros((self.num_agents, self.dim_p))
        c = jnp.zeros((self.num_agents, self.dim_c))
        
        c = c.at[0].set(actions[SPEAKER])
        
        u_act = actions[LISTENER]
        
        u_act = jnp.array([
            u_act[1] - u_act[2],
            u_act[3] - u_act[4]
        ]) * params.accel[1]
        u = u.at[1].set(u_act)

        return u, c

    
    def rewards(self, state: TargetState, params: EnvParams) -> Dict[str, float]:
        r =  -1 * jnp.sum(jnp.square(state.p_pos[1] - state.p_pos[state.goal+self.num_agents]))
        return {a: r for a in self.agents}
    
    def get_obs(self, state: TargetState, params: EnvParams):
        
        goal_colour = jnp.full((3,), 0.15)
        goal_colour = goal_colour.at[state.goal].set(0.65)        
        
        dist = state.p_pos[self.num_agents:] - state.p_pos[1]
        comm = state.c[0]
        
        def _speaker():
            return goal_colour 
        
        def _listener():
            return jnp.concatenate([
                state.p_vel[1],
                dist.flatten(),
                comm
            ])
            
        return {SPEAKER: _speaker(), LISTENER: _listener()}
    
    