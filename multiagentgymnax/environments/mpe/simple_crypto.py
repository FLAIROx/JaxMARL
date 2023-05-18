import jax 
import jax.numpy as jnp
import chex
from typing import Tuple, Dict
from flax import struct
from functools import partial
from multiagentgymnax.environments.mpe.simple import SimpleMPE, State, EnvParams
from multiagentgymnax.environments.mpe.default_params import AGENT_COLOUR, ADVERSARY_COLOUR, AGENT_RADIUS, LANDMARK_RADIUS, ADVERSARY_RADIUS, MASS, DT, MAX_STEPS, CONTACT_FORCE, CONTACT_MARGIN, ACCEL, MAX_SPEED, DAMPING  
from gymnax.environments.spaces import Box

SPEAKER = "alice_0"
LISTENER = "bob_0"
ADVERSARY = "eve_0"
SPEAKER_IDX = 1
ADVERSARY_NAMES = ["eve_0"]
OBS_COLOUR = jnp.array([[255, 0, 0, 0], [0, 255, 0, 0]])

@struct.dataclass
class CryptoState(State):
    """ State for the simple crypto environment. """
    goal_colour: chex.Array
    private_key: chex.Array

class SimpleCryptoMPE(SimpleMPE):
    """ 
    JAX Compatible version of simple_crypto_v2 PettingZoo environment.
    Source code: https://github.com/Farama-Foundation/PettingZoo/blob/master/pettingzoo/mpe/simple_crypto/simple_crypto.py 
    Note, currently only have continuous actions implemented.
    """

    def __init__(self,
                 num_agents=3,
                 num_landmarks=2,):
        
        assert num_agents == 3, "Simple Crypto only supports 3 agents"
        assert num_landmarks == 2, "Simple Crypto only supports 2 landmarks"
        
        dim_c = 4  # Communication channel dimension

        num_landmarks = num_landmarks 

        self.num_good_agents, self.num_adversaries = 2, 1
        self.num_agents = num_agents
        self.adversaries = [ADVERSARY]
        self.good_agents = [SPEAKER, LISTENER]
        
        assert self.num_agents == (self.num_good_agents + self.num_adversaries)
        assert len(self.adversaries) == self.num_adversaries
        assert len(self.good_agents) == self.num_good_agents
        
        agents = self.adversaries + self.good_agents
        assert agents[SPEAKER_IDX] == "alice_0"
        
        landmarks = ["landmark {}".format(i) for i in range(num_landmarks)]

        # Action and observation spaces
        action_spaces = {i: Box(0.0, 1.0, (4,)) for i in agents}

        observation_spaces = {i: Box(-jnp.inf, jnp.inf, (4,)) for i in self.adversaries }
        observation_spaces.update({i: Box(-jnp.inf, jnp.inf, (8,)) for i in self.good_agents})

        colour = [ADVERSARY_COLOUR] * self.num_adversaries + [AGENT_COLOUR] * self.num_good_agents + \
            list(OBS_COLOUR)
        
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
            rad=jnp.concatenate([jnp.full((self.num_adversaries), ADVERSARY_RADIUS),
                            jnp.full((self.num_good_agents), AGENT_RADIUS),
                            jnp.full((self.num_landmarks), LANDMARK_RADIUS)]),
            moveable= jnp.full((self.num_entities), False),
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
    
    def reset_env(self, key: chex.PRNGKey, params: EnvParams) -> Tuple[chex.Array, CryptoState]:
        
        key_a, key_l, key_g, key_k = jax.random.split(key, 4)        
        
        p_pos = jnp.concatenate([
            jax.random.uniform(key_a, (self.num_agents, 2), minval=-1, maxval=+1),
            jax.random.uniform(key_l, (self.num_landmarks, 2), minval=-0.9, maxval=+0.9)
        ])
        
        g_idx = jax.random.randint(key_g, (1,), minval=0, maxval=self.num_landmarks)
        k_idx = jax.random.randint(key_k, (1,), minval=0, maxval=self.num_landmarks)
        
        state = CryptoState(
            p_pos=p_pos,
            p_vel=jnp.zeros((self.num_entities, self.dim_p)),
            c=jnp.zeros((self.num_agents, self.dim_c)),
            done=jnp.full((self.num_agents), False),
            step=0,
            goal_colour=jnp.array(OBS_COLOUR[g_idx], dtype=jnp.float32).flatten(),  # set to float to be same as zoo env
            private_key=jnp.array(OBS_COLOUR[k_idx], dtype=jnp.float32).flatten(),
        )
        
        return self.get_obs(state, params), state

    @partial(jax.vmap, in_axes=[None, 0, 0, None])
    def _set_action(self, a_idx, action, params):
        """ Communication action """
        u = jnp.zeros((self.dim_p,))
        c = action
        
        return u, c

    def get_obs(self, state: CryptoState, params: EnvParams):

        goal_colour = state.goal_colour
        comm = state.c[SPEAKER_IDX]
        #jax.debug.print('comm {c}', c=state.c)
        
        def _speaker():
            return jnp.concatenate([
                goal_colour,
                state.private_key,
            ])
            
        def _listener():
            return jnp.concatenate([
                state.private_key.flatten(),
                comm,
            ])
            
        def _adversary():
            return comm
    
        # NOTE not a fan of this solution but it does work...        
        obs = {
            SPEAKER: _speaker(),
            LISTENER: _listener(),
            ADVERSARY_NAMES[0]: _adversary()
        }
        return obs
    
    def rewards(self, state: CryptoState, params: EnvParams) -> Dict[str, float]:

        comm_diff = jnp.sum(jnp.square(state.c - state.goal_colour), axis=1) # check axis
        
        comm_zeros = ~jnp.all(state.c==0)  # Ensure communication has happend
        #jax.debug.print('comm z {z}', z=comm_zeros)
        
        #mask = jnp.full((self.num_agents), -1)
        #mask = mask.at[0].set(1)
        #mask = mask.at[SPEAKER_IDX].set(0)
        mask = jnp.array([1, 0, -1])
        mask *= comm_zeros

        #jax.debug.print('mask {m}', m=mask)
        #jax.debug.print('comm diff {c}', c=comm_diff)
        def _good():
            return jnp.sum(comm_diff * mask) 
        
        def _adversary(idx):
            return -1 * jnp.sum(comm_diff[idx]) * comm_zeros

        rew = {a: _adversary(i) for i, a in enumerate(self.adversaries)}
        rew.update({a: _good() for i, a in enumerate(self.good_agents)})
        return rew

