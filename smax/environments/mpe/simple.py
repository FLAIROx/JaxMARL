""" 
Base env class for MPE PettingZoo envs.

NOTE: only for continuous action spaces currently

TODO: discrete action spaces
TODO: viz for communication env, e.g. crypto
"""

import jax
import jax.numpy as jnp 
import numpy as onp
from smax.environments.multi_agent_env import MultiAgentEnv
from smax.environments.mpe.default_params import *
import chex
from gymnax.environments.spaces import Box, Discrete
from flax import struct
from typing import Tuple, Optional, Dict
from functools import partial

@struct.dataclass
class State:
    """ Basic MPE State """
    p_pos: chex.Array # [num_entities, [x, y]]
    p_vel: chex.Array # [n, [x, y]]
    c: chex.Array  # communication state [num_agents, [dim_c]]
    done: chex.Array # bool [num_agents, ]
    step: int # current step
    
@struct.dataclass
class TargetState(State):
    """ MPE State with goal """
    goal: int  # index of target landmark

@struct.dataclass
class EnvParams:
    """ MPE Environment Parameters """
    max_steps: int  # max number of steps in an episode
    rad: chex.Array  # entity radii
    moveable: chex.Array  # true for entities that can move
    silent: chex.Array  # true for entities that cannot communicate
    collide: chex.Array  # true for entities that can collide
    mass: chex.Array  # mass of entities
    accel: chex.Array  # sensitivity, actions are multiplied by this
    max_speed: chex.Array  # max speed of entities
    u_noise: chex.Array  # physical action noise, set to 0 if no noise
    c_noise: chex.Array  # communication action noise, set to 0 if no noise
    damping: float  
    contact_force: float  
    contact_margin: float
    dt: float  # time step length

class SimpleMPE(MultiAgentEnv):
    
    def __init__(self, 
                 num_agents=1, 
                 action_type=DISCRETE_ACT,
                 agents=None,
                 num_landmarks=1,
                 landmarks=None,
                 action_spaces=None,
                 observation_spaces=None,
                 colour=None,
                 dim_c=0,
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
        self.classes = self.create_agent_classes()

        if landmarks is None:
            self.landmarks = [f"landmark {i}" for i in range(num_landmarks)]
        else:
            assert len(landmarks) == num_landmarks, f"Number of landmarks {len(landmarks)} does not match number of landmarks {num_landmarks}"
            self.landmarks = landmarks
        self.l_to_i = {l: i+self.num_agents for i, l in enumerate(self.landmarks)}

        if action_spaces is None:
            if action_type == DISCRETE_ACT:
                self.action_spaces = {i: Discrete(5) for i in self.agents}
            elif action_type == CONTINUOUS_ACT:
                self.action_spaces = {i: Box(0.0, 1.0, (5,)) for i in self.agents}
        else:
            assert len(action_spaces.keys()) == num_agents, f"Number of action spaces {len(action_spaces.keys())} does not match number of agents {num_agents}"
            self.action_spaces = action_spaces
            
        if observation_spaces is None:
            self.observation_spaces = {i: Box(-jnp.inf, jnp.inf, (4,)) for i in self.agents}
        else:
            assert len(observation_spaces.keys()) == num_agents, f"Number of observation spaces {len(observation_spaces.keys())} does not match number of agents {num_agents}"
            self.observation_spaces = observation_spaces
        
        self.colour = colour if colour is not None else [AGENT_COLOUR] * num_agents + [OBS_COLOUR] * num_landmarks
        
        # Action type
        if action_type == DISCRETE_ACT:
            self.action_decoder = self._decode_discrete_action
        elif action_type == CONTINUOUS_ACT:
            self.action_decoder = self._decode_continuous_action
        else:
            raise NotImplementedError(f"Action type: {action_type} is not supported")
        
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
            max_steps=MAX_STEPS,
            rad=jnp.concatenate([jnp.full((self.num_agents), 0.15),
                            jnp.full((self.num_landmarks), 0.2)]), # landmarks size?
            moveable=jnp.concatenate([jnp.full((self.num_agents), True), jnp.full((self.num_landmarks), False)]),
            silent = jnp.full((self.num_agents), 1),
            collide = jnp.full((self.num_entities), False),
            mass=jnp.full((self.num_entities), 1),
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

    @partial(jax.jit, static_argnums=[0])
    def step_env(self, key: chex.PRNGKey, state: State, actions: dict, params: EnvParams):
        
        u, c = self.set_actions(actions, params)
        if c.shape[1] < self.dim_c:  # This is due to the MPE code carrying around 0s for the communication channels
            c = jnp.concatenate([c, jnp.zeros((self.num_agents, self.dim_c - c.shape[1]))], axis=1)

        key, key_w = jax.random.split(key)
        p_pos, p_vel = self._world_step(key_w, state, u, params)
        key_c = jax.random.split(key, self.num_agents)
        c = self._apply_comm_action(key_c, c, params.c_noise, params.silent)
        done = jnp.full((self.num_agents), state.step>=params.max_steps)
        
        state = state.replace(
            p_pos=p_pos,
            p_vel=p_vel,
            c=c,
            done=done,
            step=state.step+1,
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
            step=0,
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
            return -1 * jnp.sum(jnp.square(state.p_pos[aidx] - state.p_pos[self.num_agents:]))
        
        r = _reward(self.agent_range, state)
        return {agent: r[i] for i, agent in enumerate(self.agents)}
            
    def set_actions(self, actions: Dict, params: EnvParams):
        """ Extract u and c actions for all agents from actions Dict."""
        
        actions = jnp.array([actions[i] for i in self.agents]).reshape((self.num_agents, -1))

        return self.action_decoder(self.agent_range, actions, params)

    '''@partial(jax.vmap, in_axes=[None, 0, 0, None])
    def _set_action(self, a_idx: int, action: chex.Array, params: EnvParams) -> Tuple[chex.Array, chex.Array]:
        # NOTE only for continuous action space currently
        u = jnp.array([
            action[1] - action[2],
            action[3] - action[4]
        ])
        
        #print('params moveable', params.moveable[a_idx])
        u = u * params.accel[a_idx] * params.moveable[a_idx]
        c = action[5:] 
        return u, c'''
    
    @partial(jax.vmap, in_axes=[None, 0, 0, None])
    def _decode_continuous_action(self, a_idx:int, action: chex.Array, params: EnvParams) -> Tuple[chex.Array, chex.Array]:
        u = jnp.array([
            action[1] - action[2],
            action[3] - action[4]
        ])
        
        u = u * params.accel[a_idx] * params.moveable[a_idx]
        c = action[5:] 
        return u, c
    
    @partial(jax.vmap, in_axes=[None, 0, 0, None])
    def _decode_discrete_action(self, a_idx:int, action: chex.Array, params: EnvParams) -> Tuple[chex.Array, chex.Array]:
        u = jnp.zeros((self.dim_p,))        
        idx = jax.lax.select(action <= 2, 0, 1)
        u_val = jax.lax.select(action % 2 == 0, 1.0, -1.0) * (action != 0)
        u = u.at[idx].set(u_val)  
        u = u * params.accel[a_idx] * params.moveable[a_idx] 
        return u, jnp.zeros((self.dim_c,))     
    
    # return all entities in the world
    @property
    def entities(self):
        return self.entity_range

    def _world_step(self, key: chex.PRNGKey, state: State, u: chex.Array, params: EnvParams):        
        
        p_force = jnp.zeros((self.num_agents, 2))  
        
        # apply agent physical controls
        key_noise = jax.random.split(key, self.num_agents)
        p_force = self._apply_action_force(key_noise, p_force, u, params.u_noise, params.moveable[:self.num_agents])
        
        # apply environment forces
        p_force = jnp.concatenate([p_force, jnp.zeros((self.num_landmarks, 2))])
        p_force = self._apply_environment_force(p_force, state, params)
        #print('p_force post apply env force', p_force)
        #jax.debug.print('jax p_force {p_force}', p_force=p_force)

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
        
        p_pos += p_vel * params.dt
        p_vel = p_vel * (1 - params.damping)
        
        p_vel += (p_force / mass) * params.dt * moveable
        
        speed = jnp.sqrt(
                    jnp.square(p_vel[0]) + jnp.square(p_vel[1])
        )        
        over_max = (p_vel / jnp.sqrt(jnp.square(p_vel[0]) + jnp.square(p_vel[1])) * max_speed)
        
        p_vel = jax.lax.select((speed > max_speed) & (max_speed >= 0), over_max, p_vel)
          
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
    
    def create_agent_classes(self):
        if hasattr(self, "leader"):
            return {"leadadversary": self.leader,
                    "adversaries": self.adversaries, 
                    "agents": self.good_agents,}
        elif hasattr(self, "adversaries"):
            return {"adversaries": self.adversaries, 
                    "agents": self.good_agents,}
        else:
            return {"agents": self.agents,}


    def agent_classes(self):
        return self.classes
    
    ### === UTILITIES === ###
    def is_collision(self, a:int, b:int, state: State, params: EnvParams):
        """ check if two entities are colliding """
        dist_min = params.rad[a] + params.rad[b]
        delta_pos = state.p_pos[a] - state.p_pos[b]
        dist = jnp.sqrt(jnp.sum(jnp.square(delta_pos)))
        return (dist < dist_min) & (params.collide[a] & params.collide[b]) #& (a != b)
        
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
        
        return ax.imshow(rgb_array)
    
    def update_render(self, im, state: State, params: Optional[EnvParams] = None):
        ax = im.axes 
        return self.init_render(ax, state, params)
    

if __name__=="__main__":
    from smax.viz.visualizer import Visualizer

    num_agents = 3
    key = jax.random.PRNGKey(0)
    
    env = SimpleMPE(num_agents)
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
    print('action spaces', env.action_spaces)
    
    for _ in range(50):
        state_seq.append(state)
        key, key_act = jax.random.split(key)
        key_act = jax.random.split(key_act, env.num_agents)
        actions = {agent: env.action_space(agent).sample(key_act[i]) for i, agent in enumerate(env.agents)} 
        print('actions', actions)
        
        obs, state, rew, dones, _ = env.step_env(key, state, actions, params)
        print('state', obs)
        
        #env.render(state, params)
        #raise
        #pygame.time.wait(300)

    viz = Visualizer(env, state_seq, params)
    viz.animate(view=True)