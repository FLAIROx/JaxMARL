"""
2D robot navigation simulator that follows the JaxMARL interface 
"""

import jax
import jax.numpy as jnp
from functools import partial
import chex 
from flax import struct
from typing import Tuple, Dict
import matplotlib.axes._axes as axes

from jaxmarl.environments import MultiAgentEnv
from jaxmarl.environments.spaces import Box, Discrete

from .maps import make_map, Map
from .jaxnav_utils import wrap, cart2pol

NUM_REWARD_COMPONENTS = 2
REWARD_COMPONENT_SPARSE = 0
REWARD_COMPONENT_DENSE = 1
@struct.dataclass
class Reward:
    sparse: jnp.ndarray
    dense: jnp.ndarray

def listify_reward(reward: Reward, do_batchify=False): # returns shape of (*batch, num_agents, 2)
    ans = jnp.stack(
        [reward.sparse, reward.dense],
        axis=-1
    )
    # batchify stacks the agents first and then does reshape, which is why we need the swapaxes.
    if do_batchify:
        ans = jnp.swapaxes(ans, 0, 1).reshape(-1, *ans.shape[2:]) # shape of (batch * num_agents, 2)
    return ans

@struct.dataclass
class State:
    pos: chex.Array  # [n, [x, y, theta]]
    theta: chex.Array # [n, theta]
    vel: chex.Array  # [n, [speed, omega]]
    done: chex.Array  # [n, bool] whether an agent has terminated
    term: chex.Array  # [n, bool] whether an agent acted in this step
    goal_reached: chex.Array  # [n, bool] whether an agent has reached goal
    move_term: chex.Array  # [n, bool] whether an agent has crashed
    step: int  # step count  
    ep_done: bool  # whether epsiode has terminated
    goal: chex.Array  # [n, x, y]
    map_data: chex.Array  # occupancy grid for environment map
    rew_lambda: float  # linear interpolation between individual and team rewards
    
@struct.dataclass
class EnvInstance:
    agent_pos: chex.Array
    agent_theta: chex.Array
    goal_pos: chex.Array
    map_data: chex.Array
    rew_lambda: chex.Array

### ---- Discrete action constants ----
DISCRETE_ACTS = jnp.array([
    jnp.array([0.0, 0.5]),
    jnp.array([0.0, 0.25]),
    jnp.array([0.0, 0.0]),
    jnp.array([0.0, -0.25]),
    jnp.array([0.0, -0.5]),
    jnp.array([0.5, 0.5]),
    jnp.array([0.5, 0.25]),
    jnp.array([0.5, 0.0]),
    jnp.array([0.5, -0.25]),
    jnp.array([0.5, -0.5]),
    jnp.array([1.0, 0.5]),
    jnp.array([1.0, 0.25]),
    jnp.array([1.0, 0.0]),
    jnp.array([1.0, -0.25]),
    jnp.array([1.0, -0.5]),
], dtype=jnp.float32)


@partial(jax.vmap, in_axes=[0])
def discrete_act_map(action: int) -> jnp.ndarray:
    print('action', action, action.shape)
    return DISCRETE_ACTS[action]
    
## ---- Environment defaults ----
AGENT_BASE = "agent"
MAP_PARAMS = {
    "map_size": (7, 7),
    "fill": 0.3,
}

## ---- Environment ----
class JaxNav(MultiAgentEnv):
    """ 
    Current assumptions:
     - homogenous agents
    """
        
    def __init__(self,
                 num_agents: int, # Number of agents
                 act_type="Continuous", # Action type, either Continuous or Discrete
                 normalise_obs=True,
                 rad=0.3,  # Agent radius, TODO remove dependency on this
                 evaporating=False,  # Whether agents evaporate (dissapeare) when they reach the goal
                 map_id="Grid-Rand-Poly",  # Map type
                 map_params=MAP_PARAMS,  # Map parameters
                 lidar_num_beams=200,
                 lidar_range_resolution=0.05,
                 lidar_max_range=6.0,
                 lidar_min_range=0.0,
                 lidar_angle_factor=1.0,
                 min_v=0.0,
                 max_v=1.0, 
                 max_v_acc=1.0,
                 max_w=1.0,
                 max_w_acc=1.0,
                 max_steps=500,
                 dt=0.1,
                 fixed_lambda=True,
                 rew_lambda=1.0, # linear interpolation between individual and team rewards
                 lambda_range=[0.0, 1.0],
                 goal_radius=0.3,
                 goal_rew=4.0,
                 weight_g=0.25,
                 lim_w=0.7,
                 weight_w=-0.0,
                 dt_rew=-0.01,
                 coll_rew=-5.0,
                 lidar_thresh=0.1,
                 lidar_rew=-0.1,
                 do_sep_reward=False,
                 share_only_sparse=False,
                 info_by_agent=False,
        ):
        super().__init__(num_agents)
        
        assert rad < 1, "current code assumes radius of less than 1"
        self.rad = rad
        self.agents = ["agent_{}".format(i) for i in range(num_agents)]
        self.agent_range = jnp.arange(0, num_agents)
        self.evaporating = evaporating
        
        self._map_obj = make_map(map_id, self.num_agents, self.rad, **map_params)
        self._act_type = act_type
        if self._act_type == "Discrete":
            assert min_v == 0.0, "min_v must be 0.0 for Discrete actions"
        
        # Lidar parameters
        self.normalise_obs = normalise_obs
        self.lidar_num_beams = lidar_num_beams
        self.lidar_max_range = lidar_max_range
        self.lidar_min_range = lidar_min_range
        assert self.lidar_min_range == 0.0, "lidar_min_range must be 0.0" # TODO for now
        self.lidar_range_resolution = lidar_range_resolution
        self.lidar_angle_factor = lidar_angle_factor
        self.lidar_max_angle = jnp.pi * self.lidar_angle_factor
        self.lidar_angles = jnp.linspace(-jnp.pi * self.lidar_angle_factor, jnp.pi * self.lidar_angle_factor, self.lidar_num_beams) 
        num_lidar_samples = int((self.lidar_max_range - self.lidar_min_range) / self.lidar_range_resolution)
        self.lidar_ranges = jnp.linspace(self.lidar_min_range, self.lidar_max_range, num_lidar_samples)
        
        assert min_v < max_v, "min_v must be less than max_v"
        if min_v != 0.0: print(f"WARNING: min_v is not 0.0, it is {min_v}")
        self.min_v = min_v  # min linear velocity (m/s)
        self.max_v = max_v  # max linear velocity (m/s)
        self.max_v_acc = max_v_acc  # max linear acceleration (m/s^2)
        self.max_w = max_w  # max angular velocity (rad/s)
        self.max_w_acc = max_w_acc  # max angular acceleration (rad/s^2)
        self.max_steps = max_steps  # max environment steps within an episode
        self.dt = dt  # seconds per step (s)    

        # Rewards
        # if share_only_sparse:
        #     do_sep_reward, f"If share_only_sparse is True, do_sep_reward must be True for it to work, it is current: {do_sep_reward}"
        self.do_sep_reward = do_sep_reward
        self.share_only_sparse = share_only_sparse
        self.fixed_lambda = fixed_lambda
        self.rew_lambda = rew_lambda  # linear interpolation between individual and team rewards
        self.lambda_range = lambda_range
        if self.fixed_lambda: assert self.rew_lambda is not None, "If fixed_lambda is True, rew_lambda must be set"
        self.goal_radius = goal_radius  # goal radius (m)
        self.goal_rew = goal_rew
        self.weight_g = weight_g
        self.lim_w = lim_w
        self.weight_w = weight_w
        self.dt_rew = dt_rew
        self.coll_rew = coll_rew
        self.lidar_thresh = lidar_thresh 
        self.lidar_rew = lidar_rew
        
        self.info_by_agent = info_by_agent
        self.eval_solved_rate = self.get_eval_solved_rate_fn()
        
        self.action_spaces = {a: self.agent_action_space() for a in self.agents}
        self.observation_spaces = {a: self.agent_observation_space() for a in self.agents}
        
    @property
    def map_obj(self) -> Map:
        """ Return map object """
        return self._map_obj
    
    
    @partial(jax.jit, static_argnums=[0])  
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], State]:
        """ Reset environment. Returns initial agent observations, states and the enviornment state """
        
        state = self.sample_test_case(key)
        obs = self._get_obs(state)
        return {a: obs[i] for i, a in enumerate(self.agents)}, state
        
    @partial(jax.jit, static_argnums=[0])  
    def step_env(
                self,
                key: chex.PRNGKey,
                agent_states: State,
                actions: Dict[str, chex.Array]
        ):
        actions = jnp.array([actions[a] for a in self.agents])  # Batchify
        
        # 1) Update agent states
        if self._act_type == "Discrete": actions = discrete_act_map(actions).reshape(actions.shape[0], 2)
        old_pos = agent_states.pos
        update_state_valid = agent_states.done 
        if not self.evaporating:
            update_state_valid = update_state_valid | agent_states.move_term
        new_pos, new_theta, new_vel = self.update_state(agent_states.pos, agent_states.theta, agent_states.vel, actions, update_state_valid)                      
        step = agent_states.step+1 
                        
        # 2) Check collisions, goal and time
        old_goal_reached = agent_states.goal_reached
        old_move_term = agent_states.move_term
        map_collisions = jax.vmap(self._map_obj.check_agent_map_collision, in_axes=(0, 0, None))(new_pos, new_theta, agent_states.map_data)*(1-agent_states.done).astype(bool)
        if self.num_agents > 1:
            agent_collisions = self.map_obj.check_all_agent_agent_collisions(new_pos, new_theta)*(1- agent_states.done).astype(bool)
        else:
            agent_collisions = jnp.zeros((self.num_agents,), dtype=jnp.bool_)
        collisions = map_collisions | agent_collisions
        goal_reached = (self._check_goal_reached(new_pos, agent_states.goal)*(1-agent_states.done)).astype(bool)
        time_up = jnp.full((self.num_agents,), (step >= self.max_steps))

        # 3) Compute rewards and done values
        old_done = agent_states.done
        if self.evaporating:
            dones = collisions | goal_reached | time_up | agent_states.done  # OR operation over agent status
            ep_done = jnp.all(dones) 
        else:
            goal_reached = goal_reached | old_goal_reached
            collisions = collisions | old_move_term
            ep_done = jnp.all(goal_reached | collisions | time_up)
            dones = jnp.full((self.num_agents,), ep_done)        

        # 4) Update JAX state
        agent_states = agent_states.replace(
            pos=new_pos,
            theta=new_theta,
            vel=new_vel,
            move_term=collisions,
            goal_reached=goal_reached,
            done=dones,
            term=old_done,
            step=step,  
            ep_done=ep_done,
        )
                
        dones = {a: agent_states.done[i] for i, a in enumerate(self.agents)}
        dones["__all__"] = ep_done
        
        # 5) Compute observations
        obs_batch = self._get_obs(agent_states)
        
        # 6) Reward
        rew_individual, individual_rew_sep = self.compute_reward(
            obs_batch,
            agent_states.pos,
            old_pos,
            actions,
            agent_states.goal,
            collisions,
            goal_reached,
            old_done,
            old_goal_reached,
            old_move_term
        )
        avg_rew = rew_individual.mean()
        if self.share_only_sparse:
            shared_rew = individual_rew_sep.sparse.mean()
        else:
            shared_rew = avg_rew
            
        if self.do_sep_reward:
            rew_batch = self.rew_lambda * rew_individual + (1 - self.rew_lambda) * shared_rew
        else:
            rew_batch = self.rew_lambda * rew_individual + (1 - self.rew_lambda) * shared_rew

        rew = {a: rew_batch[i] for i, a in enumerate(self.agents)}
        obs = {a: obs_batch[i] for i, a in enumerate(self.agents)}
        
        if self.evaporating:
            num_c = jnp.sum(collisions | goal_reached | time_up)
            time_o = time_up & ~old_done
        else:
            num_c = jax.lax.select(ep_done, self.num_agents, 0)
            time_o = time_up & ~(collisions | goal_reached)
            
        goal_r = goal_reached * (1 - old_goal_reached)
        agent_c = agent_collisions * (1 - old_move_term)
        map_c = map_collisions * (1 - old_move_term)
        rew_info = avg_rew
        if not self.info_by_agent:
            goal_r = jnp.sum(goal_r)
            agent_c = jnp.sum(agent_c)
            map_c = jnp.sum(map_c)
            time_o = jnp.sum(time_o)
            term = {a: old_done[i] for i, a in enumerate(self.agents)}
        else:
            num_c = jnp.full((self.num_agents,), ep_done, dtype=jnp.int32)
            rew_info = rew_batch
            term = old_done
        
        info = {
            # outcomes
            "NumC": num_c,
            "GoalR": goal_r,
            "AgentC": agent_c,
            "MapC": map_c,
            "TimeO": time_o,
            "Return": rew_info,  # reward
            "terminated": term,  # whether action was valid
        }
        if self.do_sep_reward:
            raise NotImplementedError("Separate reward not implemented")
            return obs, agent_states, individual_rew_sep, dones, info  # NOTE no sharing ..?
        else:
            return obs, agent_states, rew, dones, info
    
    def _lidar_sense(self, idx: int, state: State) -> chex.Array:
        """ Return observation for an agent given the current world state """
        
        pos = state.pos[idx]
        theta = state.theta[idx]
        
        point_fn = jax.vmap(self._map_obj.check_point_map_collision, in_axes=(0, None))
        
        angles = self.lidar_angles + theta

        angles_ranges_mesh = jnp.meshgrid(angles, self.lidar_ranges)  # value mesh
        angles_ranges = jnp.dstack(angles_ranges_mesh)  # reformat array [num_points_per_beam, num_beams, 2]
        beam_coords_x = (angles_ranges[:,:,1]*jnp.cos(angles_ranges[:,:,0])).T + pos[0]  
        beam_coords_y = (angles_ranges[:,:,1]*jnp.sin(angles_ranges[:,:,0])).T + pos[1]
        beam_coords = jnp.dstack((beam_coords_x, beam_coords_y))  # [num_beams, num_points_per_beam, 2]
    
        agent_c = self._lidar_agent_check(self.agent_range, state.pos, state.theta, beam_coords, idx)
        rc_range = jnp.where(agent_c==-1, jnp.inf, agent_c)
        rc_m = jnp.min(rc_range, axis=0)
        rc = jnp.where(rc_m==jnp.inf, -1, rc_m).astype(int)
        
        lidar_hits = point_fn(beam_coords.reshape(-1, 2), state.map_data).reshape(beam_coords.shape[0], beam_coords.shape[1], -1)
        
        idxs = jnp.arange(0, beam_coords.shape[0])
        lidar_hits = lidar_hits.at[idxs, rc].set(1)
        fh_idx = jnp.argmax(lidar_hits>0, axis=1)
        return self.lidar_ranges[fh_idx]
    
    @partial(jax.vmap, in_axes=[None, 0, 0, 0, None, None])
    def _lidar_agent_check(self, other_idx, other_pos, other_theta, beam_coords, host_idx):
        """ Compute lidar collisions with other robots, vectorised across other agent indicies 
        Returns:
            chex.Array: index of lidar particle for which the hit occured, direction to goal in global frame,
                        distance to goal
        """
        
        i = jax.vmap(
            self._map_obj.check_agent_beam_intersect,
            in_axes=(0, None, None, None)
        )(beam_coords, other_pos, other_theta, self.lidar_range_resolution)        
        return jax.lax.select(other_idx==host_idx, jnp.full(i.shape, -1), i)
        
            
    def normalise_lidar(self, ranges):
        return ranges/self.lidar_max_range - 0.5
    
    def unnormalise_lidar(self, ranges):
        return (ranges + 0.5) * self.lidar_max_range
            
    def get_avail_actions(self, state: State):
        
        return {a: jnp.array([1.0, 1.0]) for a in self.agents}  
             
    def sample_test_case(self, key: chex.PRNGKey) -> State:
        
        key_tc, key_lambda = jax.random.split(key)
        map_data, test_case = self._map_obj.sample_test_case(key_tc)
                        
        states = State(
            pos=test_case[:, 0, :2],
            theta=test_case[:, 0, 2], 
            vel=jnp.zeros((self.num_agents, 2)),
            done=jnp.full((self.num_agents,), False),
            term=jnp.full((self.num_agents,), False),  # TODO don't think this is  needed
            goal_reached=jnp.full((self.num_agents,), False),
            move_term=jnp.full((self.num_agents,), False),
            step=0,
            ep_done=False,
            goal=test_case[:, 1, :2],
            map_data=map_data,
            rew_lambda=self.sample_lambda(key_lambda),
        )
        
        return states

    @partial(jax.jit, static_argnums=[0])
    def sample_lambda(self, key):
        if self.fixed_lambda:
            rew_lambda = self.rew_lambda
        else:
            rew_lambda = jax.random.uniform(key, (1,), minval=self.lambda_range[0], maxval=self.lambda_range[1])
        return rew_lambda
    
    @partial(jax.vmap, in_axes=(None, 0, 0, None))
    def _check_map_collisions(self, pos: chex.Array, theta: chex.Array, map_data: chex.Array) -> bool:
        return self._map_obj.check_agent_map_collision(pos, theta, map_data)
       
    @partial(jax.vmap, in_axes=(None, 0, 0))
    def _check_goal_reached(self, pos: chex.Array, goal_pos: chex.Array) -> bool:
        return jnp.sqrt(jnp.sum((pos - goal_pos)**2)) <= self.goal_radius 
            
    @partial(jax.jit, static_argnums=[0])
    def get_obs(self, state: State) -> chex.Array:
        obs_batch = self._get_obs(state)
        return {a: obs_batch[i] for i, a in enumerate(self.agents)}
        
    @partial(jax.jit, static_argnums=[0])
    def _get_obs(self, state: State) -> chex.Array:
        """ Return observation for an agent given the current world state
        
        obs: [lidar (num lidar beams), speeds (2), goal (2), lambda (1)]
        """
         
        @partial(jax.vmap, in_axes=[0, None])
        def _observation(idx: int, state: State) -> jnp.ndarray:
            """Return observation for agent i."""
            
            lidar = self._lidar_sense(idx, state).squeeze()
                    
            vel_obs = state.vel[idx] 
            goal_dir = state.goal[idx] - state.pos[idx]
            goal_obs = cart2pol(*goal_dir)
            goal_dist = jnp.clip(goal_obs[0], 0, self.lidar_max_range) 
            goal_orient = wrap(goal_obs[1]-state.theta[idx])
            
            if self.normalise_obs:
                lidar = self.normalise_lidar(lidar)
                vel_obs = vel_obs / jnp.array([self.max_v, self.max_w]) - jnp.array([0.5, 0.0])
                goal_dist = goal_dist/self.lidar_max_range - 0.5
                goal_orient = goal_orient/jnp.pi
                rew_lambda = state.rew_lambda - 0.5
            vel_goal = jnp.concatenate([vel_obs, goal_dist[None], goal_orient[None], jnp.array([rew_lambda]).reshape(1)])
            return jnp.concatenate((lidar, vel_goal))
        
        return _observation(self.agent_range, state)
    
    @partial(jax.jit, static_argnums=[0])
    def get_world_state(self, state: State) -> chex.Array:
        walls = state.map_data.at[1:-1, 1:-1].get().flatten()
        pos = (state.pos / jnp.array([self._map_obj.width, self._map_obj.height]) - 0.5).flatten()
        theta = (state.theta / jnp.pi - 0.5).flatten()
        goal = (state.goal / jnp.array([self._map_obj.width, self._map_obj.height]) - 0.5).flatten()
        vel = (state.vel / jnp.array([self.max_v, self.max_w]) - 0.5).flatten()
        step = jnp.array(state.step / self.max_steps - 0.5)[None]
        concat = (jnp.concatenate([walls, pos, theta, goal, vel, step])[None]).repeat(self.num_agents, axis=0)
        agent_idx = jnp.eye(self.num_agents)
        
        obs = self._get_obs(state)
        
        return jnp.concatenate([agent_idx, concat, obs], axis=1)
    
    @partial(jax.vmap, in_axes=(None, 0, 0, 0, 0, 0))
    def update_state(self, pos: chex.Array, theta: float, speed: chex.Array, action: chex.Array, done: chex.Array) -> chex.Array:
        """ Update agent's state, if `done` the current position and velocity are returned"""
        if self.evaporating:
            out_done = (jnp.array([0.0,0.0]), theta, jnp.array([0.0,0.0]))  # "Evaporating" agents
        else:
            out_done = (pos, theta, jnp.array([0.0, 0.0]))

        # check if action within limits        
        v_acc = jnp.clip((action[0] - speed[0])/self.dt, -self.max_v_acc, self.max_v_acc)
        w_acc = jnp.clip((action[1] - speed[1])/self.dt, -self.max_w_acc, self.max_w_acc)
        
        v = jnp.clip(speed[0] + v_acc*self.dt, self.min_v, self.max_v)
        w = jnp.clip(speed[1] + w_acc*self.dt, -self.max_w, self.max_w)
        
        dx = v * jnp.cos(theta) * self.dt
        dy = v * jnp.sin(theta) * self.dt
        pos = pos + jnp.array([dx, dy])
        theta = wrap(theta + w*self.dt)
        
        out = (pos, theta, jnp.array([v, w], dtype=jnp.float32))
        return jax.tree.map(lambda x, y: jax.lax.select(done, x, y), out_done, out)  
        
    @partial(jax.vmap, in_axes=(None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    def compute_reward(
                    self,
                    obs,
                    new_pos,
                    old_pos,
                    act,
                    goal,
                    collision,
                    goal_reached,
                    done,
                    old_goal_reached,
                    old_move_term,
        ):
        rga = self.weight_g * (jnp.linalg.norm(old_pos - goal) - jnp.linalg.norm(new_pos - goal))
        rg = jnp.where(goal_reached, self.goal_rew, rga)  * (1 - old_goal_reached)# goal reward
        rc = collision * self.coll_rew * (1 - old_move_term) # collision reward
        rw = jax.lax.select(jnp.abs(act[1]) > self.lim_w, self.weight_w * jnp.abs(act[1]), 0.0)  # angular velocity magnitue penalty
        rt = self.dt_rew * (1 - (old_goal_reached | old_move_term))  # time penalty
        rl = jnp.any(self.unnormalise_lidar(obs[:self.lidar_num_beams]) <= (self.lidar_thresh + self.rad)) * self.lidar_rew # lidar proximity reward 

        ret = Reward((jnp.where(goal_reached, rg, 0.0) + rc + rt)*(1 - done),
                     (jnp.where(goal_reached, 0.0, rg) + rw + rl)*(1 - done))

        # {
        #     # Sparse reward is goal reward if it was reached & collision reward.
        #     'sparse': jnp.where(goal_reached, rg, 0.0) + rc + rt,
        #     # 
        #     'dense': jnp.where(goal_reached, 0.0, rg) + rw + rl,
        # }

        return (rg + rc + rw + rt + rl)*(1 - done), ret
    
    def set_state(
        self,
        state: State
    ) -> Tuple[Dict[str, chex.ArrayTree], State]:
        """
        Implemented for basic envs.
        """
        obs = self._get_obs(state)
        return {a: obs[i] for i, a in enumerate(self.agents)}, state
        
    def set_env_instance(
        self,
        encoding: EnvInstance
    ) -> Tuple[Dict[str, chex.ArrayTree], State]:
        """
        Instance is encoded as a PyTree containing the following fields:
        agent_pos, agent_theta, goal_pos, map_data
        """
        state = State(
            pos=encoding.agent_pos,
            theta=encoding.agent_theta,
            vel=jnp.zeros((self.num_agents, 2)),
            done=jnp.full((self.num_agents,), False),
            term=jnp.full((self.num_agents,), False),
            goal_reached=jnp.full((self.num_agents,), False),
            move_term=jnp.full((self.num_agents,), False),
            step=0,
            ep_done=False,
            goal=encoding.goal_pos,
            map_data=encoding.map_data,
            rew_lambda=encoding.rew_lambda
        )
        obs = self._get_obs(state)
        return {a: obs[i] for i, a in enumerate(self.agents)}, state
        
    @partial(jax.jit, static_argnums=(0))
    def reset_to_level(self, level: Tuple[chex.Array, chex.Array]) -> Tuple[chex.Array, State]:
        print(' ** WARNING ** reset_to_level in JaxNav is deprecated, use set_state instead')
        map_data, test_case = level
        
        state = State(
            pos=test_case[:, 0, :2],
            theta=test_case[:, 0, 2], 
            vel=jnp.zeros((self.num_agents, 2)),
            done=jnp.full((self.num_agents,), False),
            term=jnp.full((self.num_agents,), False),
            step=0,
            ep_done=False,
            goal=test_case[:, 1, :2],
            map_data=map_data,
            rew_lambda=self.rew_lambda,
        )
        obs = self._get_obs(state)
        return {a: obs[i] for i, a in enumerate(self.agents)}, state
        
    @partial(jax.jit, static_argnums=(0,))
    def step_plr(
        self,
        key: chex.PRNGKey,
        state: State,
        actions: chex.Array,
        level: Tuple,
    ):
        """ Resets to PLR level rather than a random one."""
        print(' ** WARNING ** step_plr in JaxNav is deprecated ')
        obs_st, state_st, rewards, dones, infos = self.step_env(
            key, state, actions
        )
        obs_re, state_re = self.reset_to_level(level)  # todo maybe should be set state depending on PLR code
        state = jax.tree.map(
            lambda x, y: jax.lax.select(state_st.ep_done, x, y), state_re, state_st
        )
        obs = jax.tree.map(
            lambda x, y: jax.lax.select(state_st.ep_done, x, y), obs_re, obs_st
        )
        #obs = jax.lax.select(state_st.ep_done, obs_re, obs_st)
        return obs, state, rewards, dones, infos

    @partial(jax.jit, static_argnums=[0])
    def unnormalise_obs(self, obs_batch: chex.Array) -> chex.Array:
        lidar = self.unnormalise_lidar(obs_batch[:, :self.lidar_num_beams])
        vel_obs = (obs_batch[:, self.lidar_num_beams:self.lidar_num_beams+2] + jnp.array([0.5, 0.0])) * jnp.array([self.max_v, self.max_w]) 
        goal_dist = (obs_batch[:, self.lidar_num_beams+2:self.lidar_num_beams+3] + 0.5) * self.lidar_max_range
        goal_orient = obs_batch[:, self.lidar_num_beams+3:self.lidar_num_beams+4] * jnp.pi
        rew_lambda = obs_batch[:, -1] + 0.5
        vel_goal = jnp.concatenate([vel_obs, goal_dist, goal_orient, rew_lambda[:, None]], axis=1)
        o = jnp.concatenate([lidar, vel_goal], axis=1)
        return o
    
    def get_monitored_metrics(self):
        return ["NumC", "GoalR", "AgentC", "MapC", "TimeO", "Return"]

    def get_eval_solved_rate_fn(self):
        def _fn(ep_stats):
            return ep_stats["GoalR"] / ep_stats["NumC"]
            
        return _fn

    def agent_action_space(self):
        if self._act_type == "Discrete":
            return Discrete(15)
        low = jnp.array(
            [self.min_v, -jnp.pi/6], dtype=jnp.float32 # NOTE hard coded heading angle
        )
        high = jnp.array(
            [self.max_v, jnp.pi/6], dtype=jnp.float32 
        )
        return Box(low, high, (2,), jnp.float32)
            
    
    def agent_observation_space(self):
        return Box(-jnp.inf, jnp.inf, (self.lidar_num_beams+5,))  # NOTE hardcoded
    
    # def action_space(self, agent=None):   # NOTE assuming homogenous observation spaces, NOTE I think jnp.empty is fine
    #     aa = self.agent_action_space()
    #     return jnp.empty((self.num_agents, *aa.shape))
    
    # def observation_space(self, agent=None):
    #     oo = self.agent_observation_space()
    #     return jnp.empty((self.num_agents, *oo.shape))
    
    @partial(jax.jit, static_argnums=[0])
    def generate_scenario(self, key):
        """ Sample map grid and agent start/goal poses """
        return self._map_obj.sample_scenario(key)
    
    def get_env_metrics(self, state: State) -> dict:
        """ NOTE only valid for grid map type"""
        # n_walls = state.map_data.sum() - state.map_data.shape[0]*2 - state.map_data.shape[1]*2 + 4
        inside = state.map_data.astype(jnp.bool_)[1:-1, 1:-1]
        n_walls = jnp.sum(inside)

        passable, path_len = jax.vmap(
            self.map_obj.dikstra_path,
            in_axes=(None, 0, 0)
        )(
            state.map_data,
            state.pos,
            state.goal,
        )
                
        shortest_path_lengths_stderr = jax.lax.select(
            jnp.sum(passable) > 0,
            jnp.std(path_len, where=passable)/jnp.sqrt(jnp.sum(passable)),
            0.0
        )
        return dict(
            n_walls=n_walls,
            shortest_path_length_mean=jnp.mean(path_len, where=passable),
            shortest_path_lengths_stderr=shortest_path_lengths_stderr,
            passable=jnp.mean(passable),
        )
        
    ### === VISULISATION === ###

    def plot_lidar(self, ax: axes.Axes, obs: Dict, state: State, num_to_plot: int=10):

        @partial(jax.vmap, in_axes=(0, 0, 0, None))
        def lidar_scatter(
            pos: chex.Array,
            theta: float,
            lidar_ranges: chex.Array,
            idx,
        ):
            """Return lidar ranges as points ready to be plotted with `ax.scatter()`

            Args:
                state (EnvSingleAState): agent state
                params (EnvParams): environment parameters
                ranges (chex.Array): reported lidar ranges

            Returns:
                Tuple[List, List]: lists of x and y coordinates respectively of lidar ranges for plotting
            """
            ranges = self.unnormalise_lidar(lidar_ranges) # (lidar_ranges+0.5)*self.lidar_max_range  # correct normalisation
            x = [ranges[i]*jnp.cos(self.lidar_angles[idx[i]]+theta) + pos[0] for i in range(ranges.shape[0])]
            y = [ranges[i]*jnp.sin(self.lidar_angles[idx[i]]+theta) + pos[1] for i in range(ranges.shape[0])]
            return jnp.array([x, y])  

        if self.lidar_num_beams>10:
            if num_to_plot > self.lidar_num_beams: 
                num_to_plot = self.lidar_num_beams
                print('Warning: num_to_plot > lidar_num_beams, setting num_to_plot to lidar_num_beams')
            idx = jnp.round(jnp.linspace(0, self.lidar_num_beams-1, num_to_plot)).astype(int)
        else:
            idx = range(self.lidar_num_beams)

        obs_batch = jnp.stack([obs[a] for a in self.agents])

        lidar_scat = lidar_scatter(state.pos, state.theta, obs_batch[:, idx], idx)
        lidar_scat = jnp.swapaxes(lidar_scat, 1, 2).reshape((-1, 2))
        lidar_scat = self._map_obj.scale_coords(lidar_scat)
        ax.scatter(lidar_scat[:, 0], lidar_scat[:, 1], c='b',  s=2)

    # Plotting by SMAX style
    def init_render(self, 
                    ax: axes.Axes, 
                    state: State, 
                    obs: Dict=None, 
                    lidar=True,  # plot lidar?
                    agent=True,  # plot agents?
                    goal=True,  # plot goals?
                    rew_lambda=False,  # plot lambda?
                    ticks_off=False,  # turn off axis ticks?
                    num_to_plot=10,  # number of lidar beams to plot
                    colour_agents_by_idx=False,
                    ):
        """ Render environment. """
 
        ax.set_aspect('equal', 'box')
        
        self.map_obj.plot_map(ax, state.map_data)
        if agent: 
            self.map_obj.plot_agents(ax, state.pos, state.theta, state.goal, state.done, plot_line_to_goal=goal, colour_agents_by_idx=colour_agents_by_idx)
        if lidar: 
            assert obs is not None, "lidar is True but no obs provided, TODO make it not obs dependent"
            self.plot_lidar(ax, obs, state, num_to_plot=num_to_plot)
        
        if rew_lambda:
            ax.text(0.5, 0.5, f"lambda: {state.rew_lambda}", fontsize=12, ha='center', va='center', c='white')
        if ticks_off:
            ax.set_xticks([])
            ax.set_yticks([])

        canvas = ax.figure.canvas
        canvas.draw()

    
