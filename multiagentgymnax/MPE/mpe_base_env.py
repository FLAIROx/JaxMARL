""" 
Base env class for MPE PettingZoo envs.
"""

import jax
import jax.numpy as jnp 
import numpy as onp
from multiagentgymnax import MultiAgentEnv
import chex
import pygame
"""
has both a physical and communication action

agents change each step - state
landmarks are constant - params

need to deal with action call back


do we need params. like no we do not, brax just uses state
gymnax uses params

NOTE only continuous actions fo rnow

key qs
- how to id agents, I think just index makes the most sense
- how to deal with different action spaces... 


landmarks.. 
can be added to list of agents?


"""

from flax import struct
from typing import Tuple, Optional
from functools import partial
import os
@struct.dataclass
class State:
    p_pos: chex.Array # [n, [x, y]]
    p_vel: chex.Array # [n, [x, y]]
    s_c: chex.Array # communication state
    u: chex.Array # physical action
    c: chex.Array  # communication action

@struct.dataclass
class MPEParams:
    
    a_rad: chex.Array
    a_moveable: chex.Array
    a_collide: chex.Array
    
    u_noise: chex.Array  # set to 0 if no noise
    c_noise: chex.Array
    
    l_pos: chex.Array 
    l_rad: chex.Array
    l_moveable: chex.Array
    


class SimpleMPEEnv(MultiAgentEnv):
    
    def __init__(self, num_agents, num_landmarks=1):
        # list of agents and entities (can change at execution-time!)
        #self.agents = []
        self.num_agents = num_agents
        self.num_landmarks = num_landmarks
        self.num_entities = num_agents + num_landmarks
        
        # NOTE this should all be initalised with a config file
        self.rad = jnp.concatenate([jnp.full((num_agents), 0.1),
                                    jnp.full((num_landmarks), 0.2)])
        self.moveable = jnp.concatenate([jnp.full((num_agents), True),
                                           jnp.full((num_landmarks), False)])
        self.collide = jnp.full((self.num_entities), True)
        self.mass = jnp.full((self.num_entities), 1)
        self.max_speed = jnp.concatenate([jnp.full((num_agents), 0.5),
                                          jnp.full((num_landmarks), 0.0)])
        self.a_colour = [(225, 225, 0)] * num_agents + [(225, 225, 255)] * num_landmarks
        
        self.u_noise = jnp.concatenate([jnp.full((num_agents), 1),  # set to 0 if no noise
                                        jnp.full((num_landmarks), 0)])
        self.c_noise = jnp.concatenate([jnp.full((num_agents), 1),  # set to 0 if no noise
                                        jnp.full((num_landmarks), 0)])
        
        #self.l_pos: chex.Array 
        #self.l_rad: chex.Array
        #self.l_moveable: chex.Array
        
        
        self.agent_range = jnp.arange(num_agents)
        self.entity_range = jnp.arange(self.num_entities)
        
        # communication channel dimensionality
        self.dim_c = 3
        # position dimensionality
        self.dim_p = 2
        # color dimensionalit
        # y
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.1
        # physical damping
        self.damping = 0.25
        # contact response parameters
        self.contact_force = 1e2
        self.contact_margin = 1e-3

        # PLOTTING
        self.render_mode = "human"
        self.width = 700
        self.height = 700
        self.screen = pygame.Surface([self.width, self.height])
        self.max_size = 1
        '''self.game_font = pygame.freetype.Font(
            os.path.join(os.path.dirname(__file__), "secrcode.ttf"), 24
        )'''
        self.renderOn = False

    def step_env(self, key, state, actions):
        
        u, c = self._set_action(self.agent_range, actions)
        print('u ', u, 'c', c)
        
        p_pos, p_vel, c = self.world_step(key, state, u)
        
        state = State(
            p_pos=p_pos,
            p_vel=p_vel,
            s_c=state.s_c,
            u=u,
            c=c,
        )
        return state
        
        # TODO Rewards
        
        
    def reset_env(self):
        
        state = State(
        p_pos=jnp.array([[1.0, 1.0], [0.0, 0.0], [-1.0, 0.0], [0.5, 0.5]]),
        p_vel=jnp.zeros((self.num_entities, 2)),
        s_c=jnp.zeros((self.num_entities, 2)),
        u=jnp.zeros((self.num_entities, 2)),
        c=jnp.zeros((self.num_entities, 2)),
        )
        
        return state
        
    @partial(jax.vmap, in_axes=[None, 0, 0])
    def _set_action(self, a_idx, action):
        
        #u = jnp.zeros(self.dim_p)
        #u[0] += action[0][1] - action[0][2]
        #u[1] += action[0][3] - action[0][4]
        
        u = jnp.array([
            action[0][1] - action[0][2],
            action[0][3] - action[0][4]
        ])
        
        # TODO add sensitivity
        # TODO add moveable & silence 
        
        c = action[1]
        return u, c

    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks




    def world_step(self, key, state, u):
        
        p_force = jnp.zeros((self.num_agents, 2))  # TODO entities
        
        # apply agent physical controls
        key_noise = jax.random.split(key, self.num_entities)
        p_force = self.apply_action_force(key_noise, p_force, u, self.u_noise, self.moveable)
        print('p_force post apply action force', p_force)
        
        # apply environment forces
        p_force = self.apply_environment_force(p_force, state)
        print('p_force post apply env force', p_force)
        
        # integrate physical state
        p_pos, p_vel = self.integrate_state(p_force, state.p_pos, state.p_vel, self.mass, self.max_speed)
        
        # c = self.comm_action() TODO
        c = None
        return p_pos, p_vel, c
        
        
    def comm_action(self, key, c, c_noise, silent):
        silence = jnp.zeros(c.shape)
        noise = jax.random.normal(key, shape=c.shape) * c_noise
        return jax.lax.select(silent, c + noise, silence)
        
    # gather agent action forces
    @partial(jax.vmap, in_axes=[None, 0, 0, 0, 0, 0])
    def apply_action_force(self, key, p_force, u, u_noise, moveable):
        noise = jax.random.normal(key, shape=u.shape) * u_noise * 0.0 #NOTE temp
        return jax.lax.select(moveable, u + noise, p_force)

    # gather physical forces acting on entities
    #@partial(jax.vmap, in_axes=[None, 0, 0])
    def apply_environment_force(self, p_force_all, state):
        # could do this with two vmaps
        # but the has outputs affecting all agents, thus must return a pforce of num_agent size
        
        @partial(jax.vmap, in_axes=[0, 0])
        def _env_force_outer(idx, p_force):
            
            @partial(jax.vmap, in_axes=[None, None, 0, 0])
            def _env_force_inner(idx_a, p_force_a, idx_b, p_force_b):

                l = idx_b <= idx_a 
                l_a = jnp.zeros((2, 2))
                
                cf = self.get_collision_force(idx_a, idx_b, state) 
                return jax.lax.select(l, l_a, cf)
            
            p_force_t = _env_force_inner(idx, p_force, self.agent_range, p_force_all)

            #print('p force t s', p_force_t.shape)

            p_force_a = jnp.sum(p_force_t[:, 0], axis=0)  # ego force from other agents
            p_force_o = p_force_t[:, 1]
            p_force_o = p_force_o.at[idx].set(p_force_a)
            #print('p force a', p_force_o.shape)

            return p_force_o
        
        p_forces = _env_force_outer(self.agent_range, p_force_all)
        p_forces = jnp.sum(p_forces, axis=0)  
            
        return p_forces + p_force_all
        
        # [force_a with every agent, force from a on every agent]
        
        
    # integrate physical state
    @partial(jax.vmap, in_axes=[None, 0, 0, 0, 0, 0])
    def integrate_state(self, p_force, p_pos, p_vel, mass, max_speed):
        # TODO add moveable constraint and max speed 
        #def _moveable():
        p_vel = p_vel * (1 - self.damping)
        
        p_vel += (p_force / mass) * self.dt
        
        speed = jnp.sqrt(
                    jnp.square(p_vel[0]) + jnp.square(p_vel[1])
        )
        over_max = (p_vel / jnp.sqrt(jnp.square(p_vel[0]) + jnp.square(p_vel[1])) * max_speed)
                    
        p_pos += p_vel * self.dt  
        return p_pos, p_vel  
        
    def update_agent_state(self, agent):
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
    def get_collision_force(self, idx_a, idx_b, state):
        
        dist_min = self.rad[idx_a] + self.rad[idx_b]
        delta_pos = state.p_pos[idx_a] - state.p_pos[idx_b]
                
        dist = jnp.sqrt(jnp.sum(jnp.square(delta_pos)))
        k = self.contact_margin
        penetration = jnp.logaddexp(0, -(dist - dist_min) / k) * k
        force = self.contact_force * delta_pos / dist * penetration
        force_a = +force * self.moveable[idx_a]
        force_b = -force * self.moveable[idx_b]
        force = jnp.array([force_a, force_b])
            
        
        c = ~self.collide[idx_a] |  ~self.collide[idx_b]
        c_force = jnp.zeros((2, 2)) # TODO update dimensions
        return jax.lax.select(c, c_force, force)
        
    
    def enable_render(self, mode="human"):
        if not self.renderOn and mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode(self.screen.get_size())
            self.renderOn = True 
        
        
    def render(self, state):

        self.enable_render(self.render_mode)

        self.draw(state)
        pygame.display.flip()
        return
        
        
    def draw(self, state):
        # clear screen
        self.screen.fill((255, 255, 255))

        # update bounds to center around agent
        all_poses =  state.p_pos #[entity.state.p_pos for entity in self.world.entities]
        cam_range = jnp.max(jnp.abs(jnp.array(all_poses)))

        # update geometry and text positions
        text_line = 0
        #for e, entity in enumerate(self.world.entities):
        for i in range(self.num_agents):
            # geometry
            x, y = state.p_pos[i]
            y *= (
                -1
            )  # this makes the display mimic the old pyglet setup (ie. flips image)
            x = (
                (x / cam_range) * self.width // 2 * 0.9
            )  # the .9 is just to keep entities from appearing "too" out-of-bounds
            y = (y / cam_range) * self.height // 2 * 0.9
            x += self.width // 2
            y += self.height // 2
            print(' x y', (x, y))
            pygame.draw.circle(
                self.screen, (255, 255, 0), (float(x), float(y)), float(self.rad[i]) * 350  # NOTE colour argument changed
            )  # 350 is an arbitrary scale factor to get pygame to render similar sizes as pyglet
            pygame.draw.circle(
                self.screen, (0, 0, 0), (float(x), float(y)), float(self.rad[i]) * 350, 1
            )  # borders
            assert (
                0 < x < self.width and 0 < y < self.height
            ), f"Coordinates {(x, y)} are out of bounds."

            # text
            '''if isinstance(entity, Agent):
                if entity.silent:
                    continue
                if np.all(entity.state.c == 0):
                    word = "_"
                elif self.continuous_actions:
                    word = (
                        "[" + ",".join([f"{comm:.2f}" for comm in entity.state.c]) + "]"
                    )
                else:
                    word = alphabet[np.argmax(entity.state.c)]

                message = entity.name + " sends " + word + "   "
                message_x_pos = self.width * 0.05
                message_y_pos = self.height * 0.95 - (self.height * 0.05 * text_line)
                self.game_font.render_to(
                    self.screen, (message_x_pos, message_y_pos), message, (0, 0, 0)
                )
                text_line += 1'''

    def close(self):
        if self.renderOn:
            pygame.event.pump()
            pygame.display.quit()
            self.renderOn = False

"""

how to store agent 
if it is changing, must be in the data struct. should we use index or dictionary
index seems more intuitive but jax can also vmap over a dictionary right


"""
    

if __name__=="__main__":
    
    num_agents = 3
    key = jax.random.PRNGKey(0)
    
    env = SimpleMPEEnv(num_agents)
    state = env.reset_env()
    
    
    mock_action = jnp.array([[1.0, 1.0, 0.1, 0.1]])
    
    actions = jnp.repeat(mock_action[None], repeats=num_agents, axis=0)
    print('actions', actions.shape)
    
    
    
    print('state', state)
    for _ in range(5):
        state = env.step_env(key, state, actions)
        print('state', state)
        env.render(state)
        pygame.time.wait(300)