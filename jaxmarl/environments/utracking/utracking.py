import jax
from jax import numpy as jnp
import chex
from flax import struct
import numpy as np

from functools import partial
from typing import Tuple

from .traj_models import traj_models
from jaxmarl.environments.spaces import Box, Discrete
from jaxmarl.environments.multi_agent_env import MultiAgentEnv

@jax.jit
def fill_diagonal_zeros(arr):
    # at the moment I haven't found a better way to fill the diagonal with 0s of unsquarred matrices
    return arr - arr * (jnp.eye(arr.shape[0], arr.shape[1]))

@jax.jit
def batched_least_squares(pos_x, pos_y, pos_xy, z):
    """
    Predicts in a single batch the position of multiple landmarks in respect to multiple observers and observations
    """
    N = jnp.identity(3)[:-1]
    A = jnp.full((*z.shape, 3), -1, dtype=float)
    A = A.at[..., 0].set(pos_x * 2)
    A = A.at[..., 1].set(pos_y * 2)

    weights = jnp.where(z != 0, 1, 0)[..., None] # Set the weights of missing values to 0.

    b = (jnp.einsum('...ij,...ij->...j', pos_xy, pos_xy) - (z*z))[..., None]
    A_aux = jnp.linalg.inv(jnp.einsum('...ij,...ik->...jk', A*weights, A*weights+1e-6))
    A_aux = jnp.einsum('ij,...kj->...ik', N, A_aux)
    A_aux = jnp.einsum('...ij,...kj->...ik', A_aux, A*weights)
    pred = jnp.einsum('...ij,...jk->...i', A_aux, b*weights)
    return pred


@struct.dataclass
class State:
    pos: chex.Array   # [x,y,z,angle]*num_entities, physical state of entities
    vel: chex.Array  # [float]*num_entities, velocity of entities
    traj_coeffs: chex.Array  # [float]*num_entities, coefficient of linear trajectory models
    traj_intercepts: chex.Array  # [float]*num_entities, intercept of linear trajectory models
    land_pred_pos: chex.Array # [num_agents, num_landmarks, xyz], current tracking state of each agent for each landmark
    range_buffer: chex.Array # [num_agents, num_landmarks, (observer_xy, observed_range), len(buffer)], tracking buffer for each agent-landmark pair
    steps_next_land_action: chex.Array # [int]*num_landmarks, step until when the landmarks are gonna change directions
    range_buffer_head: int # head iterator of the tracking buffer
    t: int # step

class UTracking(MultiAgentEnv):
    
    traj_models = traj_models
    discrete_actions_mapping = jnp.array([-0.24, -0.12, 0, 0.12, 0.24])
    
    def __init__(
        self,
        num_agents:int,
        num_landmarks:int,
        dt:int=30,
        max_steps:int=128,
        discrete_actions:bool=True,
        agent_depth:Tuple[float, float]=(0., 0.), # defines the range of depth for spawning agents
        landmark_depth:Tuple[float, float]=(5., 20.), # defines the range of depth for spawning landmarks
        min_valid_distance:float=5., # under this distance it's considered a crash
        min_init_distance:float=100., # minimum initial distance between vehicles
        max_init_distance:float=300., # maximum initial distance between vehicles
        max_range_dist:float=800., # above this distance a landmark is lost
        prop_agent:int=30, # rpm of agent's propellor, defines the speeds for agents (30rpm is ~1m/s)
        prop_range_landmark:Tuple[int]=(0, 5, 10), # defines the possible (propellor) speeds for landmarks
        rudder_range_landmark:Tuple[float, float]=(0.05, 0.15), # defines the angle of movement change for landmarks
        dirchange_time_range_landmark:Tuple[int, int]=(5, 15), # defines how many random steps to wait for changing the landmark directions
        tracking_buffer_len:int=32, # maximum number of range observations kept for predicting the landmark positions
        range_noise_std:float=10., # standard deviation of the gaussian noise added to range measurements
        lost_comm_prob=0.1, # probability of loosing communications
        min_steps_ls:int=2, # minimum steps for collecting data and start predicting landmarks positions with least squares
        rew_pred_thr:float=10., # tracking error threshold for tracking reward
        cont_rew:bool=True, # if false, reward becomes sparse(r) (only based on thresholds), otherwise proportional to tracking error and landmark distance
        continuous_actions:bool=False, # if false, discrete actions are defined by the discrete_actions_mapping array
        pre_init_pos:bool=True, # computing the initial positions can be expensive if done on the go; to reduce the reset (and therefore step) time, precompute a bunch of possible options
        rng_init_pos:chex.PRNGKey=jax.random.PRNGKey(0), # random seed for precomputing initial distance
        pre_init_pos_len:int=100000, # how many initial positions precompute
        debug_obs:bool=False,
        infos_for_render:bool=False,
    ): 
        assert f'dt_{dt}' in traj_models['angle'].keys(), f"dt must be in {traj_models['angle'].keys()}"
        self.dt = dt
        self.traj_model = traj_models['angle'][f'dt_{dt}']
        assert f'prop_{prop_agent}' in self.traj_model.keys(), \
            f"the propulsor velocity for agents must be in {self.traj_model.keys()}"
        assert all(f'prop_{prop}' in self.traj_model.keys() for prop in prop_range_landmark), \
            f"the propulsor choices for landmarks must be in {self.traj_model.keys()}"
        
        self.max_steps     = max_steps
        self.num_agents      = num_agents
        self.num_landmarks   = num_landmarks
        self.num_entities    = num_agents + num_landmarks
        self.agents    = [f'agent_{i}' for i in range(1, num_agents+1)]
        self.landmarks = [f'landmark_{i}' for i in range(1, num_landmarks+1)]
        self.entities  = self.agents + self.landmarks

        self.discrete_actions = discrete_actions
        self.agent_depth = agent_depth
        self.landmark_depth = landmark_depth
        self.min_valid_distance = min_valid_distance
        self.min_init_distance = min_init_distance
        self.max_init_distance = max_init_distance
        self.max_range_dist = max_range_dist
        self.prop_agent = prop_agent
        self.prop_range_landmark = prop_range_landmark
        self.rudder_range_landmark = np.array(rudder_range_landmark)
        self.dirchange_time_range_landmark = dirchange_time_range_landmark
        self.tracking_buffer_len = tracking_buffer_len
        self.range_noise_std = range_noise_std
        self.lost_comm_prob = lost_comm_prob
        self.min_steps_ls = min_steps_ls
        self.rew_pred_thr = rew_pred_thr
        self.cont_rew = cont_rew
        self.continuous_actions = continuous_actions
        self.pre_init_pos = pre_init_pos
        self.pre_init_pos_len = pre_init_pos_len
        self.debug_obs = debug_obs
        self.infos_for_render = infos_for_render

        # action and obs spaces
        if self.continuous_actions:
            self.action_spaces = {i: Box(-0.24, 0.24, (1,)) for i in self.agents}
        else:
            self.action_spaces = {i: Discrete(len(self.discrete_actions_mapping)) for i in self.agents}
        self.observation_spaces = {i: Box(-jnp.inf, jnp.inf, (6*self.num_entities,)) for i in self.agents}

        # preprocess the traj models
        self.traj_model_prop = jnp.array([int(k.split('_')[1]) for k in self.traj_model])
        self.traj_model_coeffs = jnp.array([v['coeff'] for v in self.traj_model.values()])
        self.traj_model_intercepts = jnp.array([v['intercept'] for v in self.traj_model.values()])

        # trajectory model for agents
        self.vel_model  = lambda prop: jnp.where(prop==0, 0, prop*traj_models['vel']['coeff']+traj_models['vel']['intercept'])
        self.traj_coeffs_agent = jnp.repeat(self.traj_model[f'prop_{self.prop_agent}']['coeff'], self.num_agents)
        self.traj_intercepts_agent = jnp.repeat(self.traj_model[f'prop_{self.prop_agent}']['intercept'], self.num_agents)
        self.vel_agent = jnp.repeat(self.vel_model(self.prop_agent), self.num_agents)
        self.min_agent_dist = self.vel_agent[0]*self.dt # safe distance that agents should keep between them
        
        # index of the trajectory models valid for landmarks
        self.idx_valid_traj_model_landmarks = jnp.array([
            i for i,p in enumerate(self.traj_model_prop) 
            if p in self.prop_range_landmark
        ])

        # precompute a batch of initial positions if required
        if self.pre_init_pos:
            rngs = jax.random.split(rng_init_pos, self.pre_init_pos_len)
            self.pre_init_xy = jax.jit(jax.vmap(self.get_init_pos, in_axes=(0, None, None)))(rngs, self.min_init_distance, self.max_init_distance)
            self.pre_init_choice = jnp.arange(self.pre_init_pos_len)
    
    @property
    def name(self) -> str:
        """Environment name."""
        return f'utracking_{self.num_agents}v{self.num_landmarks}_mean_rew'

    @partial(jax.jit, static_argnums=0)
    def reset(self, rng):
        
        # velocity and trajectory models
        rng, _rng = jax.random.split(rng)
        idx_traj_model_landmarks = jax.random.choice(_rng, self.idx_valid_traj_model_landmarks, shape=(self.num_landmarks,))
        traj_coeffs = jnp.concatenate((
            self.traj_coeffs_agent, # traj model for agents is costant and precomputed
            self.traj_model_coeffs[idx_traj_model_landmarks] # sample correct coeff for each landmark
        ))
        traj_intercepts = jnp.concatenate((
            self.traj_intercepts_agent, 
            self.traj_model_intercepts[idx_traj_model_landmarks] # sample intercept coeff for each landmark
        ))
        vel = jnp.concatenate((
            self.vel_agent, # vel of agents is costant and precomputed
            self.vel_model(self.traj_model_prop[idx_traj_model_landmarks])
        ))
        
        # init positions
        rng, key_pos, key_agent_depth, key_land_depth, key_dir, key_dir_change = jax.random.split(rng, 6)
        if self.pre_init_pos:
            xy_pos = self.pre_init_xy[jax.random.choice(key_pos, self.pre_init_choice)]
        else:
            xy_pos = self.get_init_pos(key_pos, self.min_init_distance, self.max_init_distance)
        z = jnp.concatenate((
            jax.random.uniform(key_agent_depth, shape=(self.num_agents,), minval=self.agent_depth[0], maxval=self.agent_depth[1]),
            jax.random.uniform(key_land_depth, shape=(self.num_landmarks,), minval=self.landmark_depth[0], maxval=self.landmark_depth[1]),
        ))
        d = jax.random.uniform(key_dir, shape=(self.num_entities,), minval=0, maxval=2*np.pi) # direction
        pos = jnp.concatenate((xy_pos, z[:, np.newaxis], d[:, np.newaxis]),axis=1)
        steps_next_land_action = jax.random.randint(key_dir_change, (self.num_landmarks,), *self.dirchange_time_range_landmark)

        # init tracking buffer variables
        land_pred_pos = jnp.zeros((self.num_agents, self.num_landmarks, 3))
        range_buffer = jnp.zeros((self.num_agents, self.num_landmarks, 3, self.tracking_buffer_len)) # num_agents, num_landmarks, xy range, len(buffer)
        range_buffer_head = 0
        t = 0
        
        # first communication
        rng, key_ranges, key_comm = jax.random.split(rng, 3)
        delta_xyz, ranges_real_2d, ranges_real, ranges = self.get_ranges(key_ranges, pos)
        range_buffer, range_buffer_head, comm_drop = self.communicate(
            key_comm,
            ranges,
            pos,
            range_buffer,
            range_buffer_head
        )
        land_pred_pos = self.update_predictions(t, range_buffer, pos, ranges)

        # first observation
        obs = self.get_obs(delta_xyz, ranges, comm_drop, pos, land_pred_pos)
        if self.debug_obs:
            obs = {a:1e-3*jnp.concatenate((pos[i], delta_xyz[i].ravel(), ranges_real_2d[i].ravel())) for i, a in enumerate(self.agents)}
        obs['world_state'] = self.get_global_state(pos, vel)

        # env state
        state = State(
            pos=pos,
            vel=vel,
            traj_coeffs=traj_coeffs,
            traj_intercepts=traj_intercepts,
            land_pred_pos=land_pred_pos,
            range_buffer=range_buffer,
            steps_next_land_action=steps_next_land_action,
            range_buffer_head=range_buffer_head,
            t=t
        )
        return obs, state

    @partial(jax.jit, static_argnums=0)
    def world_step(self, rudder_actions:chex.Array, pos:chex.Array, vel:chex.Array, traj_coeffs:chex.Array, traj_intercepts:chex.Array):
        # update the angle
        angle_change = rudder_actions*traj_coeffs+traj_intercepts
        # update the x-y position (depth remains constant)
        pos = pos.at[:, -1].add(angle_change)
        pos = pos.at[:, 0].add(jnp.cos(pos[:, -1])*vel*self.dt)
        pos = pos.at[:, 1].add(jnp.sin(pos[:, -1])*vel*self.dt)
        return pos

    @partial(jax.jit, static_argnums=0)
    def step_env(self, rng:chex.PRNGKey, state:State, actions:dict):

        # preprocess actions
        agent_actions = jnp.array([actions[a] for a in self.agents])
        agent_actions = self.preprocess_actions(agent_actions)
        landmark_actions, steps_next_land_action = self.get_landmarks_actions(rng, state.steps_next_land_action, state.t)

        # update physical positions
        pos = self.world_step(
            jnp.concatenate((agent_actions, landmark_actions)),
            state.pos,
            state.vel,
            state.traj_coeffs,
            state.traj_intercepts,
        )

        # update tracking
        rng, key_ranges, key_comm = jax.random.split(rng, 3)
        delta_xyz, ranges_real_2d, ranges_real, ranges = self.get_ranges(key_ranges, pos)
        range_buffer, range_buffer_head, comm_drop = self.communicate(
            key_comm,
            ranges,
            pos,
            state.range_buffer,
            state.range_buffer_head
        )
        land_pred_pos = self.update_predictions(state.t, range_buffer, pos, ranges)

        # get global reward, done, info
        reward, done, info = self.get_rew_done_info(state.t, pos, ranges, ranges_real_2d, land_pred_pos)
        reward = {agent:reward for agent in self.agents}
        done   = {agent:done for agent in self.agents+['__all__']}

        # agents obs and global state
        obs = self.get_obs(delta_xyz, ranges, comm_drop, pos, land_pred_pos)
        if self.debug_obs:
            obs = {a:1e-4*jnp.concatenate((pos[i], delta_xyz[i].ravel(), ranges_real_2d[i].ravel())) for i, a in enumerate(self.agents)}
        obs['world_state'] = self.get_global_state(pos, state.vel)
        
        state = state.replace(
            pos=pos,
            land_pred_pos=land_pred_pos,
            steps_next_land_action=steps_next_land_action,
            range_buffer=range_buffer,
            range_buffer_head=range_buffer_head,
            t=state.t+1
        )
        return obs, state, reward, done, info

    @partial(jax.jit, static_argnums=0)
    def get_obs(self, delta_xyz, ranges, comm_drop, pos, land_pred_pos):
        # first a matrix with all the observations is created, composed by
        # the position of the agent or the relative position of other agents (comunication) and landmarks (tracking)
        # the absolute distance (ranges) is_agent, is_self features
        # [pos_x, pos_y, pos_z, dist, is_agent, is_self]*n_entities
        other_agents_dist = jnp.where(comm_drop[:, :, None], 0, delta_xyz[:, :self.num_agents]) # 0 for communication drop
        self_mask = jnp.arange(self.num_agents) == np.arange(self.num_agents)[:, np.newaxis]
        agents_rel_pos = jnp.where(self_mask[:, :, None], pos[:self.num_agents, [0,1,3]], other_agents_dist) # for self use pos_x, pos_y, angle
        lands_rel_pos = land_pred_pos - pos[:self.num_agents, None, :3] # relative distance from predicted positions
        pos_feats = jnp.concatenate((agents_rel_pos, lands_rel_pos), axis=1)
        is_agent_feat = jnp.tile(jnp.concatenate((jnp.ones(self.num_agents), jnp.zeros(self.num_landmarks))), (self.num_agents, 1))
        is_self_feat  = (jnp.arange(self.num_entities) == jnp.arange(self.num_agents)[:, np.newaxis])
        # the distance based feats are rescaled to hundreds of meters (better for NNs)
        feats = jnp.concatenate((pos_feats*1e-4 , ranges[:, :, None]*1e-4, is_agent_feat[:, :, None], is_self_feat[:, :, None]), axis=2)
        
        # than it is assigned to each agent its obs
        return {
            a:feats[i]
            for i, a in enumerate(self.agents)
        }
    

    @partial(jax.jit, static_argnums=0)
    def get_global_state(self, pos, vel):
        # state is obs, vel, is_agent for each entity
        #pos = pos.at[:, :3].multiply(1e-3) # scale to hundreds of meters
        is_agent = jnp.concatenate((jnp.ones(self.num_agents), jnp.zeros(self.num_landmarks)))
        return jnp.concatenate((
            pos*1e-4,
            vel[:, None],
            is_agent[:, None]
        ), axis=-1)

    @partial(jax.jit, static_argnums=0)
    def get_rew_done_info(self, t, pos, ranges, ranges_real_2d, land_pred_pos):
        #aggregated because reward, done and info computations share similar computations
        
        # get the prediction error per each landmark from the agent that recived the smaller range
        land_2d_pos = pos[self.num_agents:, :2]
        land_pred_2d_pos = land_pred_pos[..., :2]
        land_ranges = ranges[:, self.num_agents:]
        
        # best prediction
        land_closest_pred_agent = jnp.argmin(jnp.where(land_ranges==0, jnp.inf, land_ranges), axis=0)
        best_2d_pred = land_pred_2d_pos[land_closest_pred_agent, jnp.arange(self.num_landmarks)]
        pred_2d_err = jnp.sqrt(jnp.sum(
            (land_2d_pos - best_2d_pred)** 2,
        axis=1))
        
        # set self-distance to inf and get the smallest distances
        distances_2d = jnp.where(jnp.eye(ranges_real_2d.shape[0], ranges_real_2d.shape[1], dtype=bool), jnp.inf, ranges_real_2d)
        min_land_dist  = distances_2d[:, self.num_agents:].min(axis=0) # distance between landmarks and their closest agent
        
        # rewards
        rew_good_pred = jnp.where(
            pred_2d_err<=self.rew_pred_thr,
            0.1, # 0.1 when the prediction is good enough
            0
        ).sum() # reward for tracking beeing under threshold 
        
        rew_land_distance_old = jnp.where(
            (min_land_dist <= self.min_agent_dist*2), # agent is close enought to the landmark 
            jnp.where(
                (min_land_dist >= self.min_agent_dist), # agent is respecting the safe distance
                1, # 1 if the agent close to the landmark is respecting the safe distance
                1, # -1 if the agent is not respecting the safe distance
            ),
            jnp.maximum(-1, -1e-3*min_land_dist), # penalty proportional to the distance of the landmark, with a max of -10
        ).min()# landmark-distance-based reward enhance the agents to stay close to landmarks while respecting safety distance
        
        distance_weight = 2 # the power for giving more weight to distance landmark
        good_distance_threshold = 1e-3*self.min_agent_dist*2 # defines when agent is close enought to the landmark 
        rew_land_distance = 1 - (
            jnp.sum(jnp.maximum(1e-3*min_land_dist-good_distance_threshold, 0)**distance_weight)
            / (self.num_landmarks * good_distance_threshold**distance_weight) # normalization (max reward==1)
        )
        rew_land_distance = jnp.maximum(rew_land_distance, -1) # lower bound

        # TODO: reward for keeping a good distance with the other agents
        # agent_dist = distances_2d[:, :self.num_agents].min(axis=1) # distance between agents and all other entities
        # pen_crash = jnp.where(min_ohter_dist < self.min_valid_distance, -1, 0).min() # penalty for crashing

        rew = rew_land_distance #+ rew_good_pred
        
        #rew = jnp.where(rew==self.num_landmarks*2, rew*10, rew) # bonus for following and tracing all the landmarks
        
        done = (
            (t == self.max_steps) # maximum steps reached
            #|((min_ohter_dist < self.min_valid_distance).any()) # or crash
            #|((min_land_dist > self.max_range_dist).any()) # or lost landmark
        )
        
        # return different infos if the env is gonna be used for rendering or training
        if self.infos_for_render:
            info = {
                'rew':rew,
                'tracking_pred': best_2d_pred,
                'tracking_error': pred_2d_err,
                'land_dist': min_land_dist,
                'tracking_error_mean': pred_2d_err.mean(),
                'land_dist_mean': min_land_dist.mean(),
            }
        else:
            info = {
                'tracking_error_mean': pred_2d_err.mean(keepdims=True),
                'land_dist_mean': min_land_dist.mean(keepdims=True),
            }
            
        return rew, done, info

    @partial(jax.jit, static_argnums=0)
    def preprocess_actions(self, actions):
        if self.continuous_actions:
            return jnp.clip(actions, a_min=-0.24, a_max=0.24).squeeze()
        else:
            return self.discrete_actions_mapping[actions]   

    @partial(jax.jit, static_argnums=0)
    def get_landmarks_actions(self, rng, steps_next_land_action, t):
        # range of change of direction is 0 unti steps_next_land_action hasn't reached t
        rng, key_action, key_sign, key_next_action = jax.random.split(rng, 4)
        action_range = jnp.where(
            steps_next_land_action[:, None]==t,
            self.rudder_range_landmark,
            jnp.zeros(2)
        )
        actions = jax.random.uniform(
            key_action,
            shape=(self.num_landmarks,),
            minval=action_range[:,0],
            maxval=action_range[:,1]
        ) * jax.random.choice(key_sign, shape=(self.num_landmarks,), a=jnp.array([-1,1])) # random sign
        # sample the next step of direction change for landmarks that changed direction
        steps_next_land_action = jnp.where(
            steps_next_land_action==t,
            t + jax.random.randint(key_next_action, (self.num_landmarks,), *self.dirchange_time_range_landmark),
            steps_next_land_action
        )
        return actions, steps_next_land_action

    @partial(jax.jit, static_argnums=0)
    def get_ranges(self, rng, pos):
        # computes the real 3d and 2d ranges and defines the observed range
        rng, key_noise, key_lost = jax.random.split(rng, 3)
        delta_xyz = pos[:self.num_agents, np.newaxis, :3] - pos[:, :3]
        ranges_real_2d = jnp.sqrt(jnp.sum(
            (delta_xyz[..., :2])** 2,
        axis=2)) # euclidean distances between agents and all other entities in 2d space
        ranges_real = jnp.sqrt(jnp.sum(
            (pos[:self.num_agents, np.newaxis, :3] - pos[:, :3])** 2,
        axis=2)) # euclidean distances between agents and all other entities
        ranges = ranges_real + jax.random.normal(key_noise, shape=ranges_real.shape)*self.range_noise_std # add noise
        ranges = jnp.where(
            (jax.random.uniform(key_lost, shape=ranges.shape)>self.lost_comm_prob)|(ranges_real>self.max_range_dist), # lost communication or landmark too far
            ranges,
            0
        ) # lost communications
        ranges = fill_diagonal_zeros(ranges) # reset to 0s the self-ranges
        return delta_xyz, ranges_real_2d, ranges_real, ranges

    @partial(jax.jit, static_argnums=0)
    def communicate(self, rng, ranges, pos, range_buffer, range_buffer_head):
        
        rng, key_comm = jax.random.split(rng)
        
        # comm_drop is a bool mask that defines which agent-to-agent communications are dropped
        comm_drop = jax.random.uniform(key_comm, shape=(self.num_agents, self.num_agents))<self.lost_comm_prob
        comm_drop = fill_diagonal_zeros(comm_drop).astype(bool)
        
        # exchange landmark ranges between agents
        land_ranges = jnp.tile(ranges[:, self.num_agents:], (self.num_agents, 1, 1))
        land_ranges = jnp.where(comm_drop[..., np.newaxis], 0, land_ranges) # lost comunicatio data becomes zero
        land_ranges = jnp.swapaxes(land_ranges, 1, 2)[:, :, np.newaxis, :] # num_agents, num_landmarks, None, n_observations (num_agents)
        
        # exchange position of the observers (agents) for each range
        range_pos = jnp.tile(pos[:self.num_agents, :2], (self.num_agents, 1, 1))
        range_pos = jnp.where(comm_drop[..., np.newaxis], 0, range_pos) # lost comunicatio data becomes zero
        range_pos = jnp.swapaxes(range_pos, 1, 2) # num_agents, 2 (x,y), n_observations (num_agents)
        range_pos_x = jnp.tile(range_pos[:, 0:1], (1, self.num_landmarks, 1))[:, :, np.newaxis, :]
        range_pos_y = jnp.tile(range_pos[:, 1:2], (1, self.num_landmarks, 1))[:, :, np.newaxis, :]
        
        # update the tracking buffer
        range_buffer_head = jnp.where(range_buffer_head+self.num_agents <= self.tracking_buffer_len, range_buffer_head, 0)
        range_buffer = jax.lax.dynamic_update_slice(range_buffer, range_pos_x, (0, 0, 0, range_buffer_head))
        range_buffer = jax.lax.dynamic_update_slice(range_buffer, range_pos_y, (0, 0, 1, range_buffer_head))
        range_buffer = jax.lax.dynamic_update_slice(range_buffer, land_ranges, (0, 0, 2, range_buffer_head))
        range_buffer_head += self.num_agents
        
        return range_buffer, range_buffer_head, comm_drop

    @partial(jax.jit, static_argnums=0)
    def update_predictions(self, t, range_buffer, pos, ranges):

        # if minimum steps for predicting are not reached, tracking correspond to the actual positions of the agents
        def _dummy_pred(_):
            return jnp.tile(pos[:self.num_agents, :3], (1, 1, self.num_landmarks)).reshape(self.num_agents, self.num_landmarks, -1)

        # each agents predicts the positions of the landmarks with the ranges that recived (via measuring and communication)
        def _update(_):
            pos_x = range_buffer[:, :, 0]
            pos_y = range_buffer[:, :, 1]
            pos_xy = range_buffer[:, :, :2]
            z = range_buffer[:, :, 2]
            pred_xy = batched_least_squares(pos_x, pos_y, pos_xy, z)
            pred_z  = self.estimate_depth(pred_xy, pos, ranges)
            return jnp.concatenate((pred_xy, pred_z[..., np.newaxis]), axis=-1)

        land_pred_pos = jax.lax.cond(
            t>=self.min_steps_ls,
            true_fun=_update,
            false_fun=_dummy_pred,
            operand=None
        )

        # avoid nans for singular matrix in ls computation
        land_pred_pos = jnp.where(jnp.isnan(land_pred_pos), _dummy_pred(None), land_pred_pos)
        
        return land_pred_pos

    @partial(jax.jit, static_argnums=0)
    def estimate_depth(self, pred_xy, pos, ranges):
        # bad depth estimation using pitagora
        pos_xy = pos[:self.num_agents, :2]
        ranges = ranges[:, self.num_agents:] # ranges between agents and landmarks
        delta_xy = (pred_xy - pos_xy[:, np.newaxis])**2
        to_square = ranges**2 - delta_xy[:, :, 0] - delta_xy[:, :, 1]
        z = pos[:self.num_agents, -1:] + jnp.sqrt(jnp.where(to_square>0, to_square, 0))
        return z

    @partial(jax.jit, static_argnums=0)
    def get_init_pos(self, rng, min_init_distance, max_init_distance):
        def generate_points(carry, _):
            rng, points, i = carry
            mask = jnp.arange(self.num_entities) >= i # consider a priori valid the distances with non-done points
            def generate_point(while_state):
                rng, _ = while_state
                rng, _rng = jax.random.split(rng)
                new_point = jax.random.uniform(_rng, (2,), minval=-max_init_distance, maxval=max_init_distance)
                return rng, new_point
            def is_valid_point(while_state):
                _, point = while_state
                distances = jnp.sqrt(jnp.sum((points - point)**2, axis=-1))
                return ~ jnp.all(mask | ((distances >= min_init_distance) & (distances <= max_init_distance)))
            init_point = generate_point((rng, 0))
            rng, new_point = jax.lax.while_loop(
                cond_fun = is_valid_point,
                body_fun = generate_point,
                init_val = init_point
            )
            points = points.at[i].set(new_point)
            carry = (rng, points, i+1)
            return carry, new_point
        rng, _rng = jax.random.split(rng)
        pos = jnp.zeros((self.num_entities, 2))
        pos = pos.at[0].set(jax.random.uniform(_rng, (2,), minval=-max_init_distance, maxval=max_init_distance)) # first point
        (rng, pos, i), _ = jax.lax.scan(generate_points, (rng, pos, 1), None, self.num_entities-1)
        return pos