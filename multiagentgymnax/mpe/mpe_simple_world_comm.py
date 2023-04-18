import jax 
import jax.numpy as jnp
import chex
import pygame
from typing import Tuple, Dict
from functools import partial
from multiagentgymnax.mpe.mpe_base_env import MPEBaseEnv, State, EnvParams
from gymnax.environments.spaces import Box

# TODO leader mechanic (colour different)

# NOTE food and forests are part of world.landmarks

class SimpleWorldCommEnv(MPEBaseEnv):
    
    def __init__(self, 
                 num_good_agents=2, 
                 num_adversaries=4, 
                 num_obs=1,
                 num_food=2,
                 num_forests=2,):
        
        # Fixed parameters
        dim_c = 4
        
        # NOTE for now using continuous action space
        # leader continous actions =  [no_action, move_left, move_right, move_down, move_up, say_0, say_1, say_2, say_3]
        num_agents = num_good_agents + num_adversaries
        num_landmarks = num_obs + num_food + num_forests
        
        self.num_good_agents, self.num_adversaries = num_good_agents, num_adversaries
        self.num_obs, self.num_food, self.num_forests = num_obs, num_food, num_forests
        
        self.leader = "leadadversary_0"
        self.adversaries = ["adversary_{}".format(i) for i in range(num_adversaries-1)]
        self.good_agents = ["agent_{}".format(i) for i in range(num_good_agents)]
        agents = [self.leader] + self.adversaries + self.good_agents

        landmarks = ["landmark {}".format(i) for i in range(num_obs)] + \
                    ["food {}".format(i) for i in range(num_food)] + \
                    ["forest {}".format(i) for i in range(num_forests)]

        self.leader_map = jnp.insert(jnp.zeros((num_agents-1)), 0, 1)
        self.leader_idx = 0
        
        action_spaces = {i: Box(0.0, 1.0, (5,)) for i in agents}
        action_spaces[self.leader] = Box(0.0, 1.0, (9,))

        observation_spaces = {i: Box(-jnp.inf, jnp.inf, (34,)) for i in self.adversaries + [self.leader]}
        observation_spaces.update({i: Box(-jnp.inf, jnp.inf, (28,)) for i in self.good_agents})

        colour = [(243, 115, 115)] * num_adversaries + [(115, 243, 115)] * num_good_agents + \
            [(64, 64, 64)] * num_obs + [(39, 39, 166)] * num_food + [(153, 230, 153)] * num_forests
        
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
            rad=jnp.concatenate([jnp.full((self.num_adversaries), 0.075),
                               jnp.full((self.num_good_agents), 0.045),
                               jnp.full((self.num_obs), 0.2),
                               jnp.full((self.num_food), 0.03),
                               jnp.full((self.num_forests), 0.3)]),
            moveable=jnp.concatenate([jnp.full((self.num_agents), True), jnp.full((self.num_landmarks), False)]),
            silent = jnp.insert(jnp.ones((self.num_agents-1)), 0, 0).astype(jnp.int32),
            collide = jnp.concatenate([jnp.full((self.num_agents+self.num_obs), True),
                                   jnp.full(self.num_food+self.num_forests, False)]),
            mass=jnp.full((self.num_entities), 1),
            accel = jnp.concatenate([jnp.full((self.num_adversaries), 3.0),
                                 jnp.full((self.num_good_agents), 4.0)]),
            max_speed = jnp.concatenate([jnp.full((self.num_adversaries), 1.0),
                                 jnp.full((self.num_good_agents), 1.3),
                                 jnp.full((self.num_landmarks), 0.0)]),
            u_noise=jnp.full((self.num_agents), 0),
            c_noise=jnp.full((self.num_agents), 0),
            damping=0.25,  # physical damping
            contact_force=1e2,  # contact response parameters
            contact_margin=1e-3,
            dt=0.1,
        )
        return params

    @partial(jax.jit, static_argnums=[0])
    def reset_env(self, key: chex.PRNGKey, params: EnvParams) -> Tuple[chex.Array, State]:
        
        key_a, key_l = jax.random.split(key)        
        
        p_pos = jnp.concatenate([
            jax.random.uniform(key_a, (self.num_agents, 2), minval=-1, maxval=+1),
            jax.random.uniform(key_l, (self.num_landmarks, 2), minval=-0.9, maxval=+0.9)
        ])
        
        state = State(
            p_pos=p_pos,
            p_vel=jnp.zeros((self.num_entities, self.dim_p)),
            #s_c=jnp.zeros((self.num_entities, self.dim_c)),
            #u=jnp.zeros((self.num_agents, self.dim_p)),
            c=jnp.zeros((self.num_agents, self.dim_c)),
            done=jnp.full((self.num_agents), False),
            step=0
        )
        
        return self.observations(state, params), state
    
    def set_actions(self, actions: dict, params: EnvParams):
        
        @partial(jax.vmap, in_axes=[0, 0, None])
        def _set_u(a_idx, action, params):
            u = jnp.array([
                action[1] - action[2],
                action[3] - action[4]
            ])
            return u * params.accel[a_idx] * params.moveable[a_idx]

        lact = actions[self.leader]
        aact = jnp.array([actions[a] for a in self.adversaries])
        gact = jnp.array([actions[a] for a in self.good_agents])

        u_acts = jnp.concatenate([lact[:5][None], aact, gact])
        u = _set_u(self.agent_range, u_acts, params)

        c = jnp.zeros((self.num_agents, self.dim_c))
        c = c.at[self.leader_idx].set(lact[5:])

        return u, c
    
    def observations(self, state: State, params: EnvParams) -> dict:
        
        # good [self vel, self pos, landmark rel pos, other agent rel pos, other aget vel, self in forest]
        
        """ Returns observations of all agents """
        
        @partial(jax.vmap, in_axes=(0, None, None))
        def _in_forest(idx: int, state: State, params: EnvParams) -> chex.Array:
            """ Collision check for all forests with agent `idx` """
            dist=jnp.linalg.norm(state.p_pos[self.num_agents+self.num_obs+self.num_food:] - state.p_pos[idx], axis=1)
            dist_min=params.rad[-self.num_forests:] + params.rad[idx]
            return dist < dist_min

        @partial(jax.vmap, in_axes=(0, None, None, None))
        def _common_stats(aidx, forest, state, params):
            """ Values needed in all observations """
            
            landmark_pos = state.p_pos[self.num_agents:] - state.p_pos[aidx]  # Landmark positions in agent reference frame

            in_forest = jnp.any(forest[aidx])  # True if ego agent in forest
            same_forest = jnp.any(forest[aidx] * forest, axis=1)  # True if other and ego agent in same forest
            no_forest = jnp.all(~forest, axis=1) & ~in_forest  # True if other not in a forest and ego agent also not in a forest
            
            
            leader = aidx == self.leader_idx
            other_mask = jnp.logical_or(same_forest, no_forest) | leader  # Whether ego agent can see other agent
            
            # Zero out unseen agents with other_mask
            other_pos = (state.p_pos[:self.num_agents] - state.p_pos[aidx]) * other_mask[:, None]
            other_vel = state.p_vel[:self.num_agents] * other_mask[:, None]
            
            # use jnp.roll to remove ego agent from other_pos and other_vel arrays
            other_pos = jnp.roll(other_pos, shift=self.num_agents-aidx-1, axis=0)[:self.num_agents-1]
            other_vel = jnp.roll(other_vel, shift=self.num_agents-aidx-1, axis=0)[:self.num_agents-1]
            
            other_pos = jnp.roll(other_pos, shift=aidx, axis=0)
            other_vel = jnp.roll(other_vel, shift=aidx, axis=0)
            
            return landmark_pos, other_pos, other_vel, jnp.where(forest[aidx], 1, -1)
        
        forest = _in_forest(self.agent_range, state, params)
        landmark_pos, other_pos, other_vel, forest = _common_stats(self.agent_range, forest, state, params)
        
        # NOTE orderings taken from MPE code and some differ to their docs
        def _good(aidx):
            return jnp.concatenate([
                state.p_vel[aidx].flatten(), # 2
                state.p_pos[aidx].flatten(), # 2
                landmark_pos[aidx].flatten(), # 5, 2
                other_pos[aidx].flatten(), # 5, 2
                forest[aidx], # 2
                other_vel[aidx,-1:].flatten(), # 2
            ])
            
        def _leader():
            return jnp.concatenate([
                state.p_vel[self.leader_idx][None].flatten(),
                state.p_pos[self.leader_idx][None].flatten(),
                landmark_pos[self.leader_idx].flatten(), 
                other_pos[self.leader_idx].flatten(),
                other_vel[self.leader_idx, -2:].flatten(), # 2, 2
                forest[self.leader_idx],  # NOTE this differs to their docs
                state.c[self.leader_idx][None].flatten(), # 4
            ])
            
        def _adversary(aidx):
            return jnp.concatenate([
                state.p_vel[aidx][None].flatten(),
                state.p_pos[aidx][None].flatten(),
                landmark_pos[aidx].flatten(), 
                other_pos[aidx].flatten(),
                other_vel[aidx, -2:].flatten(),
                forest[aidx],
                state.c[self.leader_idx][None].flatten()
            ])
            
        # Format observations as a dictionary keyed by agent name
        obs = {self.leader: _leader()}
        obs.update({a: _adversary(i+1) for i, a in enumerate(self.adversaries)})
        obs.update({a: _good(i+self.num_adversaries) for i, a in enumerate(self.good_agents)})
        return obs


    def rewards(self, state, params) -> Dict[str, float]:

        @partial(jax.vmap, in_axes=[0, None, None])
        def _reward(aidx, state, params):
            return jax.lax.cond(aidx<self.num_adversaries, self.adversary_reward, self.agent_reward, *(aidx, state, params))
        
        r = _reward(self.agent_range, state, params)
        return {agent: r[i] for i, agent in enumerate(self.agents)}

    def agent_reward(self, aidx, state, params: EnvParams):
        
        @partial(jax.vmap, in_axes=(0,))
        def _bound_rew(x):
            w = x < 0.9
            m = x < 1.0
            mr = (x - 0.9) * 10
            br = jnp.min(jnp.array([jnp.exp(2* x - 2), 10]))
            
            return jax.lax.select(m, mr, br) * ~w
        
        rew = 0
        # check collision, -5 for each collision with adversary 
        ac = self._collision(state.p_pos[aidx], params.rad[aidx], state.p_pos[:self.num_adversaries], params.rad[:self.num_adversaries])
        rew -= jnp.sum(ac) * 5
        
        # check map bounds,  
        rew -= 2 * jnp.sum(_bound_rew(jnp.abs(state.p_pos[aidx])))
        
        # check food collisions
        fc = self._collision(state.p_pos[aidx], params.rad[aidx], state.p_pos[-(self.num_food+self.num_forests):-self.num_forests], params.rad[-(self.num_food+self.num_forests):-self.num_forests])
        rew += jnp.sum(fc) * 2
        
        # reward for being near food
        rew -= 0.05 * jnp.min(jnp.linalg.norm(state.p_pos[-(self.num_food+self.num_forests):-self.num_forests] - state.p_pos[aidx], axis=1))
        return rew
    
    #@partial(jax.vmap, in_axes=[None, 0, None])
    def adversary_reward(self, aidx, state, params: EnvParams):
        
        @partial(jax.vmap, in_axes=[0, 0, None, None])
        def vcollision(apos, arad, opos, orad):
            return self._collision(apos, arad, opos, orad)
        
        rew = 0
        
        rew -= 0.1 * jnp.min(jnp.linalg.norm(state.p_pos[self.num_adversaries:self.num_agents] - state.p_pos[aidx], axis=1))
        
        # for each agent, add collision bonus 
        rew += 5 * jnp.sum(vcollision(state.p_pos[self.num_adversaries:self.num_agents], params.rad[self.num_adversaries:self.num_agents], state.p_pos[:self.num_adversaries], params.rad[:self.num_adversaries]))
        return rew
        
        
    @partial(jax.vmap, in_axes=(None, None, None, 0, 0))
    def _collision(self, apos, arad, opos, orad):
        deltas = opos - apos
        size = arad + orad
        dist = jnp.sqrt(jnp.sum(deltas ** 2))
        return dist < size

def test_policy(key, state):
    # adversarys hunt the first good agent
    pos = state.p_pos[3]
        
    act = jnp.zeros((5, 9))
    
    o = pos - state.p_pos[:3]
    act = act.at[:3, 1].set(o[:, 0])
    act = act.at[:3, 3].set(o[:, 1])
        
    r = jax.random.uniform(key, (2, 9))
    act = act.at[3:].set(r)
    return act
    

if __name__=="__main__":
    from pettingzoo.mpe import simple_world_comm_v2

    ### Petting zoo env
    zoo_env = simple_world_comm_v2.parallel_env(max_cycles=25, continuous_actions=True)
    zoo_obs = zoo_env.reset()
    actions = {agent: zoo_env.action_space(agent).sample() for agent in zoo_env.agents}

    obs_space = {agent: zoo_env.observation_space(agent) for agent in zoo_env.agents}
    act_space = {agent: zoo_env.action_space(agent) for agent in zoo_env.agents}
    print('obs space', obs_space, '\n act space', act_space)
    #print('zoo obs', zoo_obs)
    key = jax.random.PRNGKey(0)

    env = SimpleWorldCommEnv()
    params = env.default_params
    
    key, key_r = jax.random.split(key)
    obs, state = env.reset_env(key_r, params)
    
    #obs = env.observation(0, state)
    #print('obs', obs.shape, obs)
    
    mock_action = jnp.array([[0.0, 0.0, 1.0, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0]])
    #
    #actions = jnp.repeat(mock_action[None], repeats=env.num_agents, axis=0).squeeze()
    #actions = {agent: mock_action for agent in env.agents}
    env.enable_render()
    
    print('state', state)
    for _ in range(50):
        key, key_a, key_s = jax.random.split(key, 3)
        #actions = test_policy(key_a, state)
        #actions = {agent: actions[i] for i, agent in enumerate(env.agents)}
        #print('actions', actions)
        #print('state', state)
        obs, state, rew, _ = env.step_env(key_s, state, actions, params)
        actions = {agent: zoo_env.action_space(agent).sample() for agent in zoo_env.agents}
        env.render(state, params)
        print('obs', [o.shape for o in obs.values()])
        raise
        #print('rew', rew)

