import jax 
import jax.numpy as jnp
import chex
from typing import Tuple, Dict
from functools import partial
from smax.environments.mpe.simple import SimpleMPE, TargetState, EnvParams
from smax.environments.mpe.default_params import *
from gymnax.environments.spaces import Box

# Obstacle Colours
COLOUR_1 = jnp.array([[0.75, 0.25, 0.25]])
COLOUR_2 = jnp.array([[0.25, 0.75, 0.25]])
COLOUR_3 = jnp.array([[0.25, 0.25, 0.75]])
OBS_COLOUR = jnp.concatenate([COLOUR_1, COLOUR_2, COLOUR_3])

class SimpleReferenceMPE(SimpleMPE):

    def __init__(self,
                 num_agents=2,
                 num_landmarks=3,
                 local_ratio=0.5):

        assert num_agents == 2, "SimpleReferenceMPE only supports 2 agents"        
        assert num_landmarks == 3, "SimpleReferenceMPE only supports 3 landmarks" 
        
        self.local_ratio = local_ratio
        
        dim_c = 10 

        agents = ["agent_{}".format(i) for i in range(num_agents)]

        landmarks = ["landmark {}".format(i) for i in range(num_landmarks)]

        # Action and observation spaces
        action_spaces = {i: Box(0.0, 1.0, (15,)) for i in agents}
        observation_spaces = {i: Box(-jnp.inf, jnp.inf, (8,)) for i in agents}
        colour = [AGENT_COLOUR] * num_agents + list(OBS_COLOUR)
        
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
            rad=jnp.concatenate([jnp.full((self.num_agents), AGENT_RADIUS),
                            jnp.full((self.num_landmarks), LANDMARK_RADIUS)]),
            moveable=jnp.concatenate([jnp.full((self.num_agents), True), jnp.full((self.num_landmarks), False)]),
            silent = jnp.full((self.num_agents), 0),
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
        
        g_idx = jax.random.randint(key_g, (2,), minval=0, maxval=self.num_landmarks)
        
        state = TargetState(
            p_pos=p_pos,
            p_vel=jnp.zeros((self.num_entities, self.dim_p)),
            c=jnp.zeros((self.num_agents, self.dim_c)),
            done=jnp.full((self.num_agents), False),
            step=0,
            goal=g_idx,
        )
        
        return self.get_obs(state, params), state

    def get_obs(self, state: TargetState, params: EnvParams):

        @partial(jax.vmap, in_axes=(0, None, None))
        def _common_stats(aidx, state, params):
            """ Values needed in all observations """
            
            landmark_pos = state.p_pos[self.num_agents:] - state.p_pos[aidx]  # Landmark positions in agent reference frame
            
            return landmark_pos

        landmark_pos = _common_stats(self.agent_range, state, params)

        def _agent(aidx):
            other_idx = (aidx + 1) % 2
            return jnp.concatenate([ 
                state.p_vel[aidx].flatten(), # 2
                landmark_pos[aidx].flatten(), # 3, 2
                OBS_COLOUR[state.goal[other_idx]].flatten(), # 3
                state.c[other_idx].flatten() # 10
            ])
        
        obs = {a: _agent(i) for i, a in enumerate(self.agents)}
        return obs
    
    def rewards(self, state: TargetState, params: EnvParams) -> Dict[str, float]:

        @partial(jax.vmap, in_axes=(0, None))
        def _agent(aidx, state):
            other_idx = (aidx + 1) % 2 
            return -1 * jnp.linalg.norm(state.p_pos[other_idx] - state.p_pos[self.num_agents + state.goal[other_idx]])

        agent_rew = _agent(self.agent_range, state)
        global_rew = jnp.sum(agent_rew)/self.num_agents
        rew = {a: global_rew * (1 - self.local_ratio) + agent_rew[i] * self.local_ratio for i, a in enumerate(self.agents)}
        return rew

