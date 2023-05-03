""" 
Base env class for MPE PettingZoo envs.
"""

import jax
import jax.numpy as jnp 
import numpy as onp
from multiagentgymnax.environments.multi_agent_env import MultiAgentEnv
import chex
from gymnax.environments.spaces import Box
"""
has both a physical and communication action

agents change each step - state
landmarks are constant - params

need to deal with action call back


NOTE only continuous actions fo rnow

TODO
landmarks currently have a velocity kept within the state which is just always zero. This should be removed.


"""

from flax import struct
from typing import Tuple, Optional, Dict
from functools import partial
import os
@struct.dataclass
class State:
    p_pos: chex.Array # [n, [x, y]]
    p_vel: chex.Array # [n, [x, y]]
    c: chex.Array  # communication state
    done: chex.Array # [bool,]
    step: int

@struct.dataclass
class EnvParams:
    max_steps: int
    rad: chex.Array
    moveable: chex.Array
    silent: chex.Array
    collide: chex.Array
    mass: chex.Array
    accel: chex.Array
    max_speed: chex.Array
    u_noise: chex.Array  # set to 0 if no noise
    c_noise: chex.Array
    damping: float
    contact_force: float
    contact_margin: float
    dt: float

AGENT_COLOUR = (115, 243, 115)
ADVERSARY_COLOUR = (243, 115, 115)
OBS_COLOUR = (64, 64, 64)

class MPEBaseEnv(MultiAgentEnv):
    
    def __init__(self, 
                 num_agents=1, 
                 agents=None,
                 num_landmarks=1,
                 landmarks=None,
                 action_spaces=None,
                 observation_spaces=None,
                 colour=None,
                 dim_c=3,
                 dim_p=2,):

        # Agent and entity constants
        self.num_agents = num_agents
        self.num_landmarks = num_landmarks
        self.num_entities = num_agents + num_landmarks
        self.agent_range = jnp.arange(num_agents)
        self.entity_range = jnp.arange(self.num_entities)
        
        # Setting, and sense checking, entity names and agent action spaces
        if agents is None:
            self.agents = [f"agent_{i}" for i in range(num_agents)]
        else:
            assert len(agents) == num_agents, f"Number of agents {len(agents)} does not match number of agents {num_agents}"
            self.agents = agents
        self.a_to_i = {a: i for i, a in enumerate(self.agents)}

        if landmarks is None:
            self.landmarks = [f"landmark {i}" for i in range(num_landmarks)]
        else:
            assert len(landmarks) == num_landmarks, f"Number of landmarks {len(landmarks)} does not match number of landmarks {num_landmarks}"
            self.landmarks = landmarks
        self.l_to_i = {l: i+self.num_agents for i, l in enumerate(self.landmarks)}

        if action_spaces is None:
            self.action_spaces = {i: Box(-1, 1, (5,)) for i in self.agents}
        else:
            # TODO some check with the names?
            assert len(action_spaces.keys()) == num_agents, f"Number of action spaces {len(action_spaces.keys())} does not match number of agents {num_agents}"
            self.action_spaces = action_spaces
        if observation_spaces is None:
            self.action_spaces = {i: Box(-jnp.inf, jnp.inf, (4,)) for i in self.agents}
        else:
            assert len(observation_spaces.keys()) == num_agents, f"Number of observation spaces {len(observation_spaces.keys())} does not match number of agents {num_agents}"
            self.observation_spaces = observation_spaces
        
        self.colour = colour if colour is not None else [(115, 243, 115)] * num_agents + [(64, 64, 64)] * num_landmarks
        
        # World dimensions
        self.dim_c = dim_c  # communication channel dimensionality
        self.dim_p = dim_p  # position dimensionality

        # PLOTTING
        self.render_mode = "human"
        self.width = 700
        self.height = 700
        self.renderOn = False

    @property
    def default_params(self) -> EnvParams:
        params = EnvParams(
            max_steps=25,
            rad=jnp.concatenate([jnp.full((self.num_agents), 0.1), jnp.full((self.num_landmarks), 0.2)]),
            moveable=jnp.concatenate([jnp.full((self.num_agents), True), jnp.full((self.num_landmarks), False)]),
            silent=jnp.full((self.num_agents), 0),
            collide=jnp.full((self.num_entities), True),
            mass=jnp.full((self.num_entities), 1),
            accel=jnp.full((self.num_agents), 5.0),
            max_speed=jnp.concatenate([jnp.full((self.num_agents), 0.5), jnp.full((self.num_landmarks), 0.0)]),
            u_noise=jnp.full((self.num_agents), 1),
            c_noise=jnp.full((self.num_agents), 1),
            damping=0.25,  # physical damping
            contact_force=1e2,  # contact response parameters
            contact_margin=1e-3,
            dt=0.1,
        )
        
        return params

    @partial(jax.jit, static_argnums=[0])
    def step_env(self, key: chex.PRNGKey, state: State, actions: dict, params: EnvParams):
        
        #a = dict(sorted(actions.items()))  # this does work with the string names

        u, c = self.set_actions(actions, params)
        if c.shape[1] < self.dim_c:  # This is due to the MPE code carrying around 0s for the communication channels
            c = jnp.concatenate([c, jnp.zeros((self.num_agents, self.dim_c - c.shape[1]))], axis=1)

        key, key_w = jax.random.split(key)
        p_pos, p_vel = self._world_step(key_w, state, u, params)
        
        key_c = jax.random.split(key, self.num_agents)
        c = self._apply_comm_action(key_c, c, params.c_noise, params.silent)
        done = jnp.full((self.num_agents), state.step>=params.max_steps)
        
        state = State(
            p_pos=p_pos,
            p_vel=p_vel,
            #s_c=state.s_c,
            #u=u,
            c=c,
            done=done,
            step=state.step+1
        )
        
        reward = self.rewards(state, params)
        
        obs = self.get_obs(state, params)

        info = {}
        
        dones = {a: done[i] for i, a in enumerate(self.agents)}
        dones.update({"__all__": jnp.all(done)})
        
        return obs, state, reward, dones, info
    
    @partial(jax.jit, static_argnums=[0])
    def reset_env(self, key: chex.PRNGKey, params: EnvParams) -> Tuple[chex.Array, State]:
        """ Initialise with random positions """
        
        key_a, key_l = jax.random.split(key)        
        
        p_pos = jnp.concatenate([
            jax.random.uniform(key_a, (self.num_agents, 2), minval=-1, maxval=+1),
            jax.random.uniform(key_l, (self.num_landmarks, 2), minval=-0.9, maxval=+0.9)
        ])
        
        state = State(
            p_pos=p_pos,
            p_vel=jnp.zeros((self.num_entities, self.dim_p)),
            c=jnp.zeros((self.num_agents, self.dim_c)),
            done=jnp.full((self.num_agents), False),
            step=0
        )
        
        return self.get_obs(state, params), state
    
    @partial(jax.jit, static_argnums=[0])
    def get_obs(self, state: State, params: EnvParams) -> dict:
        """ Return dictionary of agent observations """

        @partial(jax.vmap, in_axes=[0, None])
        def _observation(aidx: int, state: State) -> jnp.ndarray:
            """ Return observation for agent i."""
            landmark_rel_pos = state.p_pos[self.num_agents:] - state.p_pos[aidx]
            
            return jnp.concatenate([state.p_vel[aidx].flatten(),
                                    landmark_rel_pos.flatten()])

        obs = _observation(self.agent_range, state)
        return {a: obs[i] for i, a in enumerate(self.agents)}
    
     
    def rewards(self, state: State, params: EnvParams) -> Dict[str, float]:
        """ Assign rewards for all agents """
        
        @partial(jax.vmap, in_axes=[0, None])
        def _reward(aidx, state):
            return -1*jnp.linalg.norm(state.p_pos[aidx] - state.p_pos[-1])  # NOTE assumes one landmark 
        
        r = _reward(self.agent_range, state)
        return {agent: r[i] for i, agent in enumerate(self.agents)}
            
    def set_actions(self, actions: Dict, params: EnvParams):
        """ Extract u and c actions for all agents from actions Dict."""
        # NOTE only for continuous action space currently
        actions = jnp.array([actions[i] for i in self.agents]).squeeze()

        @partial(jax.vmap, in_axes=[0, 0, None])
        def _set_action(a_idx, action, params):
            u = jnp.array([
                action[1] - action[2],
                action[3] - action[4]
            ])
            #print('u', u)
            #print('params moveable', params.moveable[a_idx])
            u = u * params.accel[a_idx] * params.moveable[a_idx]
            c = action[5:] * ~params.silent[a_idx]
            return u, c

        return _set_action(self.agent_range, actions, params)

    # return all entities in the world
    @property
    def entities(self):
        return self.entity_range

    def _world_step(self, key, state, u, params: EnvParams):
        
        # could do less computation if we only update the agents and not landmarks, but not a big difference and makes code quite a bit easier
        
        p_force = jnp.zeros((self.num_agents, 2))  # TODO entities
        
        # apply agent physical controls
        key_noise = jax.random.split(key, self.num_agents)
        p_force = self._apply_action_force(key_noise, p_force, u, params.u_noise, params.moveable[:self.num_agents])
        
        # apply environment forces
        p_force = jnp.concatenate([p_force, jnp.zeros((self.num_landmarks, 2))])
        p_force = self._apply_environment_force(p_force, state, params)
        #print('p_force post apply env force', p_force)
        
        # integrate physical state
        p_pos, p_vel = self._integrate_state(p_force, state.p_pos, state.p_vel, params.mass, params.moveable, params.max_speed, params)
        
        # c = self.comm_action() TODO
        return p_pos, p_vel
        
    @partial(jax.vmap, in_axes=[None, 0, 0, 0, 0])
    def _apply_comm_action(self, key, c, c_noise, silent):
        silence = jnp.zeros(c.shape)
        noise = jax.random.normal(key, shape=c.shape) * c_noise
        return jax.lax.select(silent, silence, c + noise)
        
    # gather agent action forces
    @partial(jax.vmap, in_axes=[None, 0, 0, 0, 0, 0])
    def _apply_action_force(self, key, p_force, u, u_noise, moveable):
        
        noise = jax.random.normal(key, shape=u.shape) * u_noise * 0.0 #NOTE temp zeroing
        #print('p force shape', p_force.shape, 'noise shape', noise.shape, 'u shape', u.shape)
        return jax.lax.select(moveable, u + noise, p_force)

    def _apply_environment_force(self, p_force_all, state: State, params: EnvParams):
        """ gather physical forces acting on entities """
        
        @partial(jax.vmap, in_axes=[0, None, None])
        def __env_force_outer(idx, state, params):
            
            @partial(jax.vmap, in_axes=[None, 0, None, None])
            def __env_force_inner(idx_a, idx_b, state, params):

                l = idx_b <= idx_a 
                l_a = jnp.zeros((2, 2))
                
                collision_force = self._get_collision_force(idx_a, idx_b, state, params) 
                return jax.lax.select(l, l_a, collision_force)
            
            p_force_t = __env_force_inner(idx, self.entity_range, state, params)

            #print('p force t s', p_force_t.shape)

            p_force_a = jnp.sum(p_force_t[:, 0], axis=0)  # ego force from other agents
            p_force_o = p_force_t[:, 1]
            p_force_o = p_force_o.at[idx].set(p_force_a)
            #print('p force a', p_force_o.shape)

            return p_force_o
        
        p_forces = __env_force_outer(self.entity_range, state, params)
        p_forces = jnp.sum(p_forces, axis=0)  
        
        return p_forces + p_force_all        
        
    @partial(jax.vmap, in_axes=[None, 0, 0, 0, 0, 0, 0, None])
    def _integrate_state(self, p_force, p_pos, p_vel, mass, moveable, max_speed, params: EnvParams):
        """ integrate physical state """
        
        p_vel = p_vel * (1 - params.damping)
        
        p_vel += (p_force / mass) * params.dt * moveable
        
        speed = jnp.sqrt(
                    jnp.square(p_vel[0]) + jnp.square(p_vel[1])
        )        
        over_max = (p_vel / jnp.sqrt(jnp.square(p_vel[0]) + jnp.square(p_vel[1])) * max_speed)
        
        p_vel = jax.lax.select(speed > max_speed, over_max, p_vel)
        p_pos += p_vel * params.dt  
        return p_pos, p_vel  
        
    def _update_agent_state(self, agent): # TODO
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = jnp.zeros(self.dim_c)
        else:
            noise = (
                jnp.random.randn(*agent.action.c.shape) * agent.c_noise
                if agent.c_noise
                else 0.0
            )
            agent.state.c = agent.action.c + noise
            
            
    # get collision forces for any contact between two agents # TODO entities
    def _get_collision_force(self, idx_a, idx_b, state, params):
        
        dist_min = params.rad[idx_a] + params.rad[idx_b]
        #print('state', state)
        delta_pos = state.p_pos[idx_a] - state.p_pos[idx_b]
                
        dist = jnp.sqrt(jnp.sum(jnp.square(delta_pos)))
        k = params.contact_margin
        penetration = jnp.logaddexp(0, -(dist - dist_min) / k) * k
        force = params.contact_force * delta_pos / dist * penetration
        force_a = +force * params.moveable[idx_a]
        force_b = -force * params.moveable[idx_b]
        force = jnp.array([force_a, force_b])
            
        c = ~params.collide[idx_a] |  ~params.collide[idx_b]
        c_force = jnp.zeros((2, 2)) 
        return jax.lax.select(c, c_force, force)
    
    ### === UTILITIES === ###
    def is_collision(self, a:int, b:int, state: State, params: EnvParams):
        """ check if two entities are colliding """
        dist_min = params.rad[a] + params.rad[b]
        delta_pos = state.p_pos[a] - state.p_pos[b]
        dist = jnp.sqrt(jnp.sum(jnp.square(delta_pos)))
        return dist < dist_min
        
    @partial(jax.vmap, in_axes=(None, 0))
    def map_bounds_reward(self, x):
        """ vmap over x, y coodinates"""
        w = x < 0.9
        m = x < 1.0
        mr = (x - 0.9) * 10
        br = jnp.min(jnp.array([jnp.exp(2* x - 2), 10]))
        return jax.lax.select(m, mr, br) * ~w

    ### === PLOTTING === ### 
    def init_render(self, ax, state: State, params: Optional[EnvParams] = None):
        from matplotlib.patches import Circle
        import numpy as np
        if params is None:
            params = self.default_params

        ax_lim = 2
        ax.clear()
        ax.set_xlim([-ax_lim, ax_lim])
        ax.set_ylim([-ax_lim, ax_lim])
        for i in range(self.num_entities):
            c = Circle(state.p_pos[i], params.rad[i], color=onp.array(self.colour[i])/255)
            ax.add_patch(c)

        canvas = ax.figure.canvas
        canvas.draw()

        rgb_array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        rgb_array = rgb_array.reshape(canvas.get_width_height()[::-1] + (3,))
        print(rgb_array.shape)
        
        return ax.imshow(rgb_array)
    
    def update_render(self, im, state: State, params: Optional[EnvParams] = None):
        ax = im.axes 
        return self.init_render(ax, state, params)
"""

how to store agent 
if it is changing, must be in the data struct. should we use index or Dictionary
index seems more intuitive but jax can also vmap over a dictionary right


"""
    

if __name__=="__main__":
    from multiagentgymnax.viz.visualizer import Visualizer

    num_agents = 3
    key = jax.random.PRNGKey(0)
    
    env = MPEBaseEnv(num_agents)
    params = env.default_params

    obs, state = env.reset_env(key, params)
    
    
    mock_action = jnp.array([[1.0, 1.0, 0.1, 0.1, 0.0]])
    
    actions = jnp.repeat(mock_action[None], repeats=num_agents, axis=0).squeeze()

    actions = {agent: mock_action for agent in env.agents}
    a = env.agents
    a.reverse()
    print('a', a)
    actions = {agent: mock_action for agent in a}
    print('actions', actions)
    
    #env.enable_render()
    
    state_seq = []
    print('state', state)
    for _ in range(50):
        state_seq.append(state)
        obs, state, rew, dones, _ = env.step_env(key, state, actions, params)
        print('state', obs)
        
        #env.render(state, params)
        #raise
        #pygame.time.wait(300)

    viz = Visualizer(env, state_seq, params)
    viz.animate(view=True)