import jax 
import jax.numpy as jnp
import chex
import pygame
from functools import partial
from multiagentgymnax.mpe.mpe_base_env import MPEBaseEnv, MPEState

# TODO leader mechanic (colour different)

# NOTE food and forests are part of world.landmarks

class SimpleWorldCommEnv(MPEBaseEnv):
    
    def __init__(self, 
                 num_good_agents=2, 
                 num_adversaries=3, 
                 num_obs=1,
                 num_food=2,
                 num_forests=2,):
        
        dim_c = 4
        
        # NOTE for now using continuous action space
        # leader continous actions =  [no_action, move_left, move_right, move_down, move_up, say_0, say_1, say_2, say_3]
        num_agents = num_good_agents + num_adversaries
        num_landmarks = num_obs + num_food + num_forests
        
        self.num_good_agents, self.num_adversaries = num_good_agents, num_adversaries
        self.num_obs, self.num_food, self.num_forests = num_obs, num_food, num_forests
        
        self.leader = jnp.insert(jnp.zeros((num_agents-1)), 0, 1)
        self.leader_idx = 0
        
        rad = jnp.concatenate([jnp.full((num_adversaries), 0.075),
                               jnp.full((num_good_agents), 0.045),
                               jnp.full((num_obs), 0.2),
                               jnp.full((num_food), 0.03),
                               jnp.full((num_forests), 0.3)])
        
        silent = jnp.insert(jnp.ones((num_agents-1)), 0, 0).astype(jnp.int32)

        accel = jnp.concatenate([jnp.full((num_adversaries), 3.0),
                                 jnp.full((num_good_agents), 4.0)])
        max_speed = jnp.concatenate([jnp.full((num_adversaries), 1.0),
                                 jnp.full((num_good_agents), 1.3),
                                 jnp.full((num_landmarks), 0.0)])
        collide = jnp.concatenate([jnp.full((num_agents+num_obs), True),
                                   jnp.full(num_food+num_forests, False)])
        
        colour = [(243, 115, 115)] * num_adversaries + [(115, 243, 115)] * num_good_agents + \
            [(64, 64, 64)] * num_obs + [(39, 39, 166)] * num_food + [(153, 230, 153)] * num_forests
        
        super().__init__(num_agents, 
                         num_landmarks,
                         dim_c=dim_c,
                         rad=rad,
                         silent=silent,
                         collide=collide,
                         accel=accel,
                         max_speed=max_speed,
                         colour=colour)
        
        
    @partial(jax.jit, static_argnums=[0])
    def reset_env(self, key):
        
        key_a, key_l = jax.random.split(key)        
        
        p_pos = jnp.concatenate([
            jax.random.uniform(key_a, (self.num_agents, 2), minval=-1, maxval=+1),
            jax.random.uniform(key_l, (self.num_landmarks, 2), minval=-0.9, maxval=+0.9)
        ])
        
        state = MPEState(
            p_pos=p_pos,
            p_vel=jnp.zeros((self.num_entities, self.dim_p)),
            s_c=jnp.zeros((self.num_entities, self.dim_c)),
            u=jnp.zeros((self.num_entities, self.dim_p)),
            c=jnp.zeros((self.num_entities, self.dim_c)),
        )
        
        return state
    
        
    def observation(self, aidx, state):
        # NOTE have padded out the obs to all be the same size cause jax and differing array sizes.
        
        @partial(jax.vmap, in_axes=(0,))
        def __in_forest(idx) -> chex.Array:
            """ Returns true if agent is in forest """
            return jnp.linalg.norm(state.p_pos[-self.num_forests:] - state.p_pos[idx], axis=0) < 0.3  # NOTE forest size hardcoded
        
        landmark_pos = state.p_pos[self.num_agents:] - state.p_pos[aidx]  # Landmark positions in agent reference frame
        
        forest = __in_forest(self.agent_range)  # [num_agents, num_forests]
        in_forest = jnp.any(forest[aidx])  # True if ego agent in forest
        same_forest = jnp.any(forest[aidx] * forest, axis=1)  # True if other and ego agent in same forest
        no_forest = jnp.all(~forest, axis=1) & ~in_forest  # True if other not in a forest and ego agent not in forest
        
        leader = aidx == self.leader_idx
        other_mask = jnp.logical_or(same_forest, no_forest) | leader  
        
        other_pos = (state.p_pos[:self.num_agents] - state.p_pos[aidx]) * other_mask[:, None]
        other_vel = state.p_vel[:self.num_agents] * other_mask[:, None]
        
        # use jnp.roll to remove ego agent from other_pos and other_vel arrays
        other_pos = jnp.roll(other_pos, shift=self.num_agents-aidx-1, axis=0)[:self.num_agents-1]
        other_vel = jnp.roll(other_vel, shift=self.num_agents-aidx-1, axis=0)[:self.num_agents-1]
        
                
        def _good():
            return jnp.concatenate([
                state.p_pos[aidx].flatten(),
                state.p_vel[aidx].flatten(),
                landmark_pos.flatten(), 
                other_pos.flatten(),
                other_vel.flatten(),
                jnp.any(forest[aidx])[None],
                jnp.zeros((4))
            ])
            
        def _adversary():
            
            standard = lambda : jnp.concatenate([
                state.p_pos[aidx][None].flatten(),
                state.p_vel[aidx][None].flatten(),
                landmark_pos.flatten(), 
                other_pos.flatten(),
                other_vel.flatten(),
                jnp.any(forest[aidx])[None],
                state.s_c[self.leader_idx][None].flatten()
            ])
            
            leader = lambda : jnp.concatenate([
                state.p_pos[aidx][None].flatten(),
                state.p_vel[aidx][None].flatten(),
                landmark_pos.flatten(), 
                other_pos.flatten(),
                other_vel.flatten(),
                state.s_c[self.leader_idx][None].flatten(),
                jnp.zeros((1))
            ])
            
            return jax.lax.cond(aidx==self.leader_idx, leader, standard)
        
        return jax.lax.cond(aidx<self.num_adversaries, _adversary, _good)


def test_policy(key, obs):
    
    o = obs[:3, 20:22]
    print('o', o.shape, o)        
        
    r = jax.random.uniform(key, (2, 5))

if __name__=="__main__":
    key = jax.random.PRNGKey(0)

    env = SimpleWorldCommEnv()
    
    state = env.reset_env(key)
    
    obs = env.observation(0, state)
    print('obs', obs.shape, obs)
    
    mock_action = jnp.array([[0.0, 0.0, 1.0, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0]])
    
    actions = jnp.repeat(mock_action[None], repeats=env.num_agents, axis=0).squeeze()
    
    print('state', state)
    for _ in range(50):
        state = env.step_env(key, state, actions)
        print('state', state)
        env.draw2(state)
