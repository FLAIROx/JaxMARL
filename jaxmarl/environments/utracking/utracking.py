"""
TODO:
- the functions now require specific arguments, try to use only the env state when possible
"""

import jax
from jax import numpy as jnp
import chex
from flax import struct
import numpy as np

from functools import partial
from typing import Tuple, Dict

from .traj_models import traj_models
from .particle_filter import ParticlesState, ParticleFilter
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

    weights = jnp.where(z != 0, 1, 0)[
        ..., None
    ]  # Set the weights of missing values to 0.

    b = (jnp.einsum("...ij,...ij->...j", pos_xy, pos_xy) - (z * z))[..., None]
    A_aux = jnp.linalg.inv(
        jnp.einsum("...ij,...ik->...jk", A * weights, A * weights + 1e-6)
    )
    A_aux = jnp.einsum("ij,...kj->...ik", N, A_aux)
    A_aux = jnp.einsum("...ij,...kj->...ik", A_aux, A * weights)
    pred = jnp.einsum("...ij,...jk->...i", A_aux, b * weights)
    return pred


@struct.dataclass
class State:
    pos: chex.Array  # [x,y,z,angle]*num_entities, physical state of entities
    vel: chex.Array  # [float]*num_entities, velocity of entities
    traj_coeffs: chex.Array # [float]*num_entities, coefficient of linear trajectory models
    traj_intercepts: chex.Array # [float]*num_entities, intercept of linear trajectory models
    land_pred_pos: chex.Array # [num_agents, num_landmarks, xyz], current tracking state of each agent for each landmark
    range_buffer: chex.Array # [num_agents, num_landmarks, (observer_xy, observed_range), len(buffer)], tracking buffer for each agent-landmark pair
    range_buffer_head: int  # head iterator of the tracking buffer
    pf_state: ParticlesState | None  # state of the particle filter
    steps_next_land_action: chex.Array # [int]*num_landmarks, step until when the landmarks are gonna change directions
    t: int  # step
    cum_rew: float  # cumulative episode reward


class UTracking(MultiAgentEnv):

    traj_models = traj_models
    discrete_actions_mapping = jnp.array([-0.24, -0.12, 0, 0.12, 0.24])

    def __init__(
        self,
        num_agents: int,
        num_landmarks: int,
        dt: int = 30,
        max_steps: int = 128,
        discrete_actions: bool = True,
        agent_depth: Tuple[float, float] = (0.0, 0.0),  # defines the range of depth for spawning agents
        landmark_depth: Tuple[float, float] = (5.0, 20.0),  # defines the range of depth for spawning landmarks
        min_valid_distance: float = 5.0,  # under this distance it's considered a crash
        min_init_distance: float = 30.0,  # minimum initial distance between vehicles
        max_init_distance: float = 200.0,  # maximum initial distance between vehicles
        max_range_dist: float = 800.0,  # above this distance a landmark is lost
        prop_agent: int = 30,  # rpm of agent's propellor, defines the speeds for agents (30rpm is ~1m/s)
        prop_range_landmark: Tuple[int] = (0, 5, 10, 15, 20),  # defines the possible (propellor) speeds for landmarks (only some speeds are alid for now)
        rudder_range_landmark: Tuple[float, float] = (0.05, 0.15,),  # defines the angle of movement change for landmarks
        dirchange_time_range_landmark: Tuple[int, int] = (5, 15,),  # defines how many random steps to wait for changing the landmark directions
        tracking_method: str = "pf",  # method for tracking the landmarks positions (ls, pf)
        tracking_buffer_len: int = 32,  # maximum number of range observations kept for predicting the landmark positions
        range_noise_std: float = 10.0,  # standard deviation of the gaussian noise added to range measurements
        lost_comm_prob=0.1,  # probability of loosing communications
        min_steps_ls: int = 2,  # minimum steps for collecting data and start predicting landmarks positions with least squares
        rew_type: str = "tracking",  # type of reward (follow, tracking_threshold, tracking)
        rew_pred_thr: float = 10.0,  # tracking error threshold for tracking reward
        pre_init_pos: bool = True,  # computing the initial positions can be expensive if done on the go; to reduce the reset (and therefore step) time, precompute a bunch of possible options
        seed_init_pos: int = 0,  # random seed for precomputing initial distance
        pre_init_pos_len: int = 100000,  # how many initial positions precompute
        matrix_obs: bool = False,  # if true, the obs and state are matrices of features relative to all the entities, otherwise flattened
        debug_obs: bool = False,
        space_unit: float = 1e-3,  # unit of space for space observations (default to hundreds of meters)
        infos_for_render: bool = False,  # if true, additional infos are returned for rendering porpouses
        pf_num_particles: int = 5000,  # number of particles for the particle filter
        **pf_kwargs,  # kwargs for the particle filter
    ):
        assert (
            f"dt_{dt}" in traj_models["angle"].keys()
        ), f"dt must be in {traj_models['angle'].keys()}"
        self.dt = dt
        self.traj_model = traj_models["angle"][f"dt_{dt}"]
        assert (
            f"prop_{prop_agent}" in self.traj_model.keys()
        ), f"the propulsor velocity for agents must be in {self.traj_model.keys()}"
        assert all(
            f"prop_{prop}" in self.traj_model.keys() for prop in prop_range_landmark
        ), f"the propulsor choices for landmarks must be in {self.traj_model.keys()}"
        assert tracking_method in [
            "ls",
            "pf",
        ], "tracking method must be ls (Least Squares) or pf (Particle Filter)"
        assert rew_type in [
            "follow",
            "tracking_threshold",
            "tracking",
        ], "reward type must be 'follow', 'tracking_threshold' or 'tracking'"

        self.max_steps = max_steps
        self.num_agents = num_agents
        self.num_landmarks = num_landmarks
        self.num_entities = num_agents + num_landmarks
        self.agents = [f"agent_{i}" for i in range(1, num_agents + 1)]
        self.landmarks = [f"landmark_{i}" for i in range(1, num_landmarks + 1)]
        self.entities = self.agents + self.landmarks

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
        self.tracking_method = tracking_method
        self.tracking_buffer_len = tracking_buffer_len
        self.range_noise_std = range_noise_std
        self.lost_comm_prob = lost_comm_prob
        self.min_steps_ls = min_steps_ls
        self.rew_pred_thr = rew_pred_thr
        self.rew_type = rew_type
        self.pre_init_pos = pre_init_pos
        self.pre_init_pos_len = pre_init_pos_len
        self.matrix_obs = matrix_obs
        self.space_unit = space_unit   
        self.debug_obs = debug_obs
        self.infos_for_render = infos_for_render

        if tracking_method == "pf":
            self.tracking_buffer_len = (
                self.num_agents
            )  # pf doesn't need a buffer, the buffer in this case is used to store the communication between agents at current step
            self.pf = ParticleFilter(num_particles=pf_num_particles, **pf_kwargs)

        # action space
        if self.discrete_actions:
            self.action_spaces = {
                i: Discrete(len(self.discrete_actions_mapping)) for i in self.agents
            }
        else:
            self.action_spaces = {i: Box(-0.24, 0.24, (1,)) for i in self.agents}

        # obs space
        self.observation_spaces = {
            i: Box(-jnp.inf, jnp.inf, (6 * self.num_entities,)) for i in self.agents
        }

        # world state size
        self.world_state_size = (
            7 * self.num_entities
        )  # [pos_x, pos_y, pos_z, direction, vel, pred_error, is_agent]*n_entities

        # preprocess the traj models
        self.traj_model_prop = jnp.array(
            [int(k.split("_")[1]) for k in self.traj_model]
        )
        self.traj_model_coeffs = jnp.array(
            [v["coeff"] for v in self.traj_model.values()]
        )
        self.traj_model_intercepts = jnp.array(
            [v["intercept"] for v in self.traj_model.values()]
        )

        # trajectory model for agents
        self.vel_model = lambda prop: jnp.where(
            prop == 0,
            0,
            prop * traj_models["vel"]["coeff"] + traj_models["vel"]["intercept"],
        )
        self.traj_coeffs_agent = jnp.repeat(
            self.traj_model[f"prop_{self.prop_agent}"]["coeff"], self.num_agents
        )
        self.traj_intercepts_agent = jnp.repeat(
            self.traj_model[f"prop_{self.prop_agent}"]["intercept"], self.num_agents
        )
        self.vel_agent = jnp.repeat(self.vel_model(self.prop_agent), self.num_agents)
        self.min_agent_dist = (
            self.vel_agent[0] * self.dt
        )  # safe distance that agents should keep between them

        # index of the trajectory models valid for landmarks
        self.idx_valid_traj_model_landmarks = jnp.array(
            [
                i
                for i, p in enumerate(self.traj_model_prop)
                if p in self.prop_range_landmark
            ]
        )

        # precompute a batch of initial positions if required
        if self.pre_init_pos:
            rng = jax.random.PRNGKey(seed_init_pos)
            rngs = jax.random.split(rng, self.pre_init_pos_len)
            self.pre_init_xy = jax.jit(
                jax.vmap(self.get_init_pos, in_axes=(0, None, None))
            )(rngs, self.min_init_distance, self.max_init_distance)
            self.pre_init_choice = jnp.arange(self.pre_init_pos_len)

    @property
    def name(self) -> str:
        """Environment name."""
        return f"utracking_{self.num_agents}v{self.num_landmarks}_mean_rew"

    @partial(jax.jit, static_argnums=0)
    def reset(self, rng: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], State]:

        # velocity and trajectory models
        rng, _rng = jax.random.split(rng)
        idx_traj_model_landmarks = jax.random.choice(
            _rng, self.idx_valid_traj_model_landmarks, shape=(self.num_landmarks,)
        )
        traj_coeffs = jnp.concatenate(
            (
                self.traj_coeffs_agent,  # traj model for agents is costant and precomputed
                self.traj_model_coeffs[
                    idx_traj_model_landmarks
                ],  # sample correct coeff for each landmark
            )
        )
        traj_intercepts = jnp.concatenate(
            (
                self.traj_intercepts_agent,
                self.traj_model_intercepts[
                    idx_traj_model_landmarks
                ],  # sample intercept coeff for each landmark
            )
        )
        vel = jnp.concatenate(
            (
                self.vel_agent,  # vel of agents is costant and precomputed
                self.vel_model(self.traj_model_prop[idx_traj_model_landmarks]),
            )
        )

        # init positions
        rng, key_pos, key_agent_depth, key_land_depth, key_dir, key_dir_change = (
            jax.random.split(rng, 6)
        )
        if self.pre_init_pos:
            xy_pos = self.pre_init_xy[jax.random.choice(key_pos, self.pre_init_choice)]
        else:
            xy_pos = self.get_init_pos(
                key_pos, self.min_init_distance, self.max_init_distance
            )
        z = jnp.concatenate(
            (
                jax.random.uniform(
                    key_agent_depth,
                    shape=(self.num_agents,),
                    minval=self.agent_depth[0],
                    maxval=self.agent_depth[1],
                ),
                jax.random.uniform(
                    key_land_depth,
                    shape=(self.num_landmarks,),
                    minval=self.landmark_depth[0],
                    maxval=self.landmark_depth[1],
                ),
            )
        )
        d = jax.random.uniform(
            key_dir, shape=(self.num_entities,), minval=0, maxval=2 * np.pi
        )  # direction
        pos = jnp.concatenate((xy_pos, z[:, np.newaxis], d[:, np.newaxis]), axis=1)
        steps_next_land_action = jax.random.randint(
            key_dir_change, (self.num_landmarks,), *self.dirchange_time_range_landmark
        )

        # init tracking buffer variables
        land_pred_pos = jnp.zeros((self.num_agents, self.num_landmarks, 3))
        range_buffer = jnp.zeros(
            (self.num_agents, self.num_landmarks, 3, self.tracking_buffer_len)
        )  # num_agents, num_landmarks, xy range, len(buffer)
        range_buffer_head = 0
        t = 0

        # first communication and tracking
        rng, key_ranges, key_comm = jax.random.split(rng, 3)
        delta_xyz, ranges_real_2d, ranges_real, ranges_2d, ranges = self.get_ranges(
            key_ranges, pos
        )
        range_buffer, range_buffer_head, comm_drop = self.communicate(
            key_comm, ranges_2d, pos, range_buffer, range_buffer_head
        )

        # init particle filter state if required
        if self.tracking_method == "pf":
            # create the initial state for the particle filter from the ranges of each agent to the landmarks
            agents_pos_xy = pos[: self.num_agents, :2]
            land_ranges = ranges_2d[:, self.num_agents :]
            land_ranges = jnp.swapaxes(land_ranges, 0, 1)  # num_landmarks, num_agents
            rng_ = jax.random.split(rng, (self.num_landmarks, self.num_agents))
            pf_state = jax.vmap(
                jax.vmap(self.pf.reset, in_axes=(0, None, 0)), in_axes=(1, 0, 1)
            )(
                rng_, agents_pos_xy, land_ranges
            )  # num_agents, num_landmarks
        else:
            pf_state = None

        rng, rng_ = jax.random.split(rng)
        pf_state, land_pred_pos = self.update_predictions(
            rng_, pf_state, t, range_buffer, pos, ranges_2d
        )

        # first observation
        obs = self.get_obs(delta_xyz, ranges, comm_drop, pos, land_pred_pos)
        if self.debug_obs:
            obs = {
                a: self.space_unit
                * jnp.concatenate(
                    (pos[i], delta_xyz[i].ravel(), ranges_real_2d[i].ravel())
                )
                for i, a in enumerate(self.agents)
            }
        obs["world_state"] = self.get_global_state(pos, vel, ranges, land_pred_pos)

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
            pf_state=pf_state,
            t=t,
            cum_rew=0.0,
        )
        return obs, state

    @partial(jax.jit, static_argnums=0)
    def world_step(
        self,
        rudder_actions: chex.Array,
        pos: chex.Array,
        vel: chex.Array,
        traj_coeffs: chex.Array,
        traj_intercepts: chex.Array,
    ) -> chex.Array:
        # update the angle
        angle_change = rudder_actions * traj_coeffs + traj_intercepts
        # update the x-y position (depth remains constant)
        pos = pos.at[:, -1].add(angle_change)
        pos = pos.at[:, 0].add(jnp.cos(pos[:, -1]) * vel * self.dt)
        pos = pos.at[:, 1].add(jnp.sin(pos[:, -1]) * vel * self.dt)
        return pos

    @partial(jax.jit, static_argnums=0)
    def step_env(
        self, rng: chex.PRNGKey, state: State, actions: dict
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool]]:

        # preprocess actions
        agent_actions = jnp.array([actions[a] for a in self.agents])
        agent_actions = self.preprocess_actions(agent_actions)
        landmark_actions, steps_next_land_action = self.get_landmarks_actions(
            rng, state.steps_next_land_action, state.t
        )

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
        delta_xyz, ranges_real_2d, ranges_real, ranges_2d, ranges = self.get_ranges(
            key_ranges, pos
        )
        range_buffer, range_buffer_head, comm_drop = self.communicate(
            key_comm, ranges_2d, pos, state.range_buffer, state.range_buffer_head
        )
        rng, rng_ = jax.random.split(rng)
        pf_state, land_pred_pos = self.update_predictions(
            rng_, state.pf_state, state.t, range_buffer, pos, ranges_2d
        )

        # get global reward, done, info
        reward, done, info = self.get_rew_done_info(
            state.t, pos, ranges, ranges_real_2d, land_pred_pos, state.cum_rew
        )
        rewards = {agent: reward for agent in self.agents}
        done = {agent: done for agent in self.agents + ["__all__"]}

        # agents obs and global state
        obs = self.get_obs(delta_xyz, ranges, comm_drop, pos, land_pred_pos)
        if self.debug_obs:
            obs = {
                a: self.space_unit
                * jnp.concatenate(
                    (pos[i], delta_xyz[i].ravel(), ranges_real_2d[i].ravel())
                )
                for i, a in enumerate(self.agents)
            }
        obs["world_state"] = self.get_global_state(
            pos, state.vel, ranges, land_pred_pos
        )

        state = state.replace(
            pos=pos,
            land_pred_pos=land_pred_pos,
            steps_next_land_action=steps_next_land_action,
            range_buffer=range_buffer,
            range_buffer_head=range_buffer_head,
            pf_state=pf_state,
            t=state.t + 1,
            cum_rew=state.cum_rew + reward,
        )
        return obs, state, rewards, done, info

    @partial(jax.jit, static_argnums=0)
    def get_obs(
        self,
        delta_xyz: chex.Array,
        ranges: chex.Array,
        comm_drop: chex.Array,
        pos: chex.Array,
        land_pred_pos: chex.Array,
    ) -> Dict[str, chex.Array]:
        # first a matrix with all the observations is created, composed by
        # the position of the agent or the relative position of other agents (comunication) and landmarks (tracking)
        # the absolute distance (ranges) is_agent, is_self features
        # [pos_x, pos_y, pos_z, dist, is_agent, is_self]*n_entities
        other_agents_dist = jnp.where(
            comm_drop[:, :, None], 0, delta_xyz[:, : self.num_agents]
        )  # 0 for communication drop
        self_mask = (
            jnp.arange(self.num_agents) == np.arange(self.num_agents)[:, np.newaxis]
        )
        agents_rel_pos = jnp.where(
            self_mask[:, :, None], pos[: self.num_agents, [0, 1, 3]], other_agents_dist
        )  # for self use pos_x, pos_y, angle
        lands_rel_pos = (
            land_pred_pos - pos[: self.num_agents, None, :3]
        )  # relative distance from predicted positions
        pos_feats = jnp.concatenate((agents_rel_pos, lands_rel_pos), axis=1)
        is_agent_feat = jnp.tile(
            jnp.concatenate((jnp.ones(self.num_agents), jnp.zeros(self.num_landmarks))),
            (self.num_agents, 1),
        )
        is_self_feat = (
            jnp.arange(self.num_entities) == jnp.arange(self.num_agents)[:, np.newaxis]
        )
        # the distance based feats are rescaled to hundreds of meters (better for NNs)
        feats = jnp.concatenate(
            (
                pos_feats * self.space_unit,
                ranges[:, :, None] * self.space_unit,
                is_agent_feat[:, :, None],
                is_self_feat[:, :, None],
            ),
            axis=2,
        )

        # than it is assigned to each agent its obs
        return {
            a: feats[i] if self.matrix_obs else feats[i].ravel()
            for i, a in enumerate(self.agents)
        }

    @partial(jax.jit, static_argnums=0)
    def get_global_state(
        self,
        pos: chex.Array,
        vel: chex.Array,
        ranges: chex.Array,
        land_pred_pos: chex.Array,
    ) -> chex.Array:
        # state is obs, vel, is_agent, pred_error for each entity
        # pos = pos.at[:, :3].multiply(self.space_unit) # scale to hundreds of meters
        is_agent = jnp.concatenate(
            (jnp.ones(self.num_agents), jnp.zeros(self.num_landmarks))
        )

        land_2d_pos = pos[self.num_agents :, :2]
        land_pred_2d_pos = land_pred_pos[..., :2]
        land_ranges = ranges[:, self.num_agents :]

        # best prediction
        land_closest_pred_agent = jnp.argmin(
            jnp.where(land_ranges == 0, jnp.inf, land_ranges), axis=0
        )
        best_2d_pred = land_pred_2d_pos[
            land_closest_pred_agent, jnp.arange(self.num_landmarks)
        ]
        pred_2d_err = jnp.sqrt(jnp.sum((land_2d_pos - best_2d_pred) ** 2, axis=1))

        preds = jnp.concatenate(
            (jnp.zeros(self.num_agents), pred_2d_err)
        )  # 0 pred error for agents

        state = jnp.concatenate(
            (pos * self.space_unit, vel[:, None], preds[:, None] * self.space_unit, is_agent[:, None]),
            axis=-1,
        )

        if self.matrix_obs:
            return state
        else:
            return state.ravel()

    @partial(jax.jit, static_argnums=0)
    def get_rew_done_info(
        self,
        t: int,
        pos: chex.Array,
        ranges: chex.Array,
        ranges_real_2d: chex.Array,
        land_pred_pos: chex.Array,
        cum_rew: chex.Array,
    ) -> Tuple[chex.Array, chex.Array, Dict]:
        """Aggregated because reward, done and info computations share similar computations"""

        # get the prediction error per each landmark from the agent that recived the smaller range
        land_2d_pos = pos[self.num_agents :, :2]
        land_pred_2d_pos = land_pred_pos[..., :2]
        land_ranges = ranges[:, self.num_agents :]

        # best prediction is the prediction of the closest agent (i.e. the agent that received the smallest range)
        # if the range is 0, it means the range was lost, and the agent is considered to be at inf distance
        land_closest_pred_agent = jnp.argmin(
            jnp.where(land_ranges == 0, jnp.inf, land_ranges), axis=0
        )
        best_2d_pred = land_pred_2d_pos[
            land_closest_pred_agent, jnp.arange(self.num_landmarks)
        ]
        pred_2d_err = jnp.sqrt(jnp.sum((land_2d_pos - best_2d_pred) ** 2, axis=1))

        # set self-distance to inf and get the smallest distances
        distances_2d = jnp.where(
            jnp.eye(ranges_real_2d.shape[0], ranges_real_2d.shape[1], dtype=bool),
            jnp.inf,
            ranges_real_2d,
        )
        min_land_dist = distances_2d[:, self.num_agents :].min(
            axis=0
        )  # distance between landmarks and their closest agent

        agent_dist = distances_2d[:, : self.num_agents].min(
            axis=1
        )  # distance between agents and other agents

        failed_episode = (agent_dist < self.min_valid_distance).any() | (
            (min_land_dist >= self.max_range_dist).any()
        )  # failed episode if crash or lost landmark

        # rewards
        if self.rew_type == "follow":
            # reward based on percentage of landmarks followed
            rew = (min_land_dist <= self.min_agent_dist * 2).sum() / self.num_landmarks
            rew = jnp.where(
                failed_episode, -cum_rew, rew
            )  # if failed episode, episode reward goes to 0

        elif self.rew_type == "tracking_threshold":
            # reward based on percentage of landmarks correctly tracked
            rew = (pred_2d_err <= self.rew_pred_thr).sum() / self.num_landmarks
            rew = jnp.where(
                failed_episode, -cum_rew, rew
            )  # if failed episode, episode reward goes to 0

        elif self.rew_type == "tracking":
            # reward based on the tracking error
            # exponential decay of the reward based on the tracking error
            # goes to 0 if the tracking error is above the threshold
            # goes to 1 if the tracking error is 0
            thr_start_val = 0.05
            rew_fun = lambda x: jnp.exp(
                -(-jnp.log(thr_start_val) / self.rew_pred_thr) * x
            )
            rew = rew_fun(pred_2d_err.sum())
            rew = jnp.where(
                failed_episode, -cum_rew, rew
            )  # if failed episode, current episode negative reward is extended to the max steps

        elif self.rew_type == "tracking_error":
            # reward based on the tracking error
            rew = -self.space_unit * pred_2d_err.sum()
            rew = jnp.where(
                failed_episode, cum_rew + (cum_rew * self.max_steps - t), rew
            )  # if failed episode, current episode negative reward is extended to the max steps

        done = jnp.logical_or(
            t == self.max_steps,  # max steps reached
            failed_episode,  # failed episode if crash or lost landmark
        )

        # return different infos if the env is gonna be used for rendering or training
        info = {
            "landmarks_covered": (min_land_dist <= self.min_agent_dist * 2).sum(
                keepdims=True
            )
            / self.num_landmarks,
            "landmarks_lost": (min_land_dist >= self.max_range_dist).sum(keepdims=True)
            / self.num_landmarks,
            "tracking_error_mean": pred_2d_err.mean(keepdims=True),
            "land_dist_mean": min_land_dist.mean(keepdims=True),
            "crash": (agent_dist < self.min_valid_distance).any(keepdims=True),
        }
        if self.infos_for_render:
            info["render"] = {
                "pos": pos,
                "done": done,
                "reward": rew,
                "tracking_pred": best_2d_pred,
                "tracking_error": pred_2d_err,
                "land_dist": min_land_dist,
            }

        return rew, done, info

    @partial(jax.jit, static_argnums=0)
    def preprocess_actions(self, actions: chex.Array) -> chex.Array:
        if self.discrete_actions:
            return self.discrete_actions_mapping[actions]
        else:
            return jnp.clip(actions, a_min=-0.24, a_max=0.24).squeeze()

    @partial(jax.jit, static_argnums=0)
    def get_landmarks_actions(
        self, rng: chex.Array, steps_next_land_action: chex.Array, t: int
    ) -> Tuple[chex.Array, chex.Array]:
        # range of change of direction is 0 unti steps_next_land_action hasn't reached t
        rng, key_action, key_sign, key_next_action = jax.random.split(rng, 4)
        action_range = jnp.where(
            steps_next_land_action[:, None] == t,
            self.rudder_range_landmark,
            jnp.zeros(2),
        )
        actions = jax.random.uniform(
            key_action,
            shape=(self.num_landmarks,),
            minval=action_range[:, 0],
            maxval=action_range[:, 1],
        ) * jax.random.choice(
            key_sign, shape=(self.num_landmarks,), a=jnp.array([-1, 1])
        )  # random sign
        # sample the next step of direction change for landmarks that changed direction
        steps_next_land_action = jnp.where(
            steps_next_land_action == t,
            t
            + jax.random.randint(
                key_next_action,
                (self.num_landmarks,),
                *self.dirchange_time_range_landmark,
            ),
            steps_next_land_action,
        )
        return actions, steps_next_land_action

    @partial(jax.jit, static_argnums=0)
    def get_ranges(
        self, rng: chex.Array, pos: chex.Array
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
        # computes the real 3d and 2d ranges and defines the observed range
        rng, key_noise, key_lost = jax.random.split(rng, 3)
        delta_xyz = pos[: self.num_agents, np.newaxis, :3] - pos[:, :3]
        ranges_real_2d = jnp.sqrt(
            jnp.sum((delta_xyz[..., :2]) ** 2, axis=2)
        )  # euclidean distances between agents and all other entities in 2d space
        ranges_real = jnp.sqrt(
            jnp.sum((pos[: self.num_agents, np.newaxis, :3] - pos[:, :3]) ** 2, axis=2)
        )  # euclidean distances between agents and all other entities
        noise = (
            jax.random.normal(key_noise, shape=ranges_real.shape) * self.range_noise_std
        )
        ranges_2d = ranges_real_2d + noise
        ranges = ranges_real + noise
        ranges = jnp.where(
            (jax.random.uniform(key_lost, shape=ranges.shape) > self.lost_comm_prob)
            | (
                ranges_real > self.max_range_dist
            ),  # lost communication or landmark too far
            ranges,
            0,
        )  # lost communications
        ranges = fill_diagonal_zeros(ranges)  # reset to 0s the self-ranges
        return delta_xyz, ranges_real_2d, ranges_real, ranges_2d, ranges

    @partial(jax.jit, static_argnums=0)
    def communicate(
        self,
        rng: chex.Array,
        ranges: chex.Array,
        pos: chex.Array,
        range_buffer: chex.Array,
        range_buffer_head: chex.Array,
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:

        rng, key_comm = jax.random.split(rng)

        # comm_drop is a bool mask that defines which agent-to-agent communications are dropped
        comm_drop = (
            jax.random.uniform(key_comm, shape=(self.num_agents, self.num_agents))
            < self.lost_comm_prob
        )
        comm_drop = fill_diagonal_zeros(comm_drop).astype(bool)

        # exchange landmark ranges between agents
        land_ranges = jnp.tile(ranges[:, self.num_agents :], (self.num_agents, 1, 1))
        land_ranges = jnp.where(
            comm_drop[..., np.newaxis], 0, land_ranges
        )  # lost comunicatio data becomes zero
        land_ranges = jnp.swapaxes(land_ranges, 1, 2)[
            :, :, np.newaxis, :
        ]  # num_agents, num_landmarks, None, n_observations (num_agents)

        # exchange position of the observers (agents) for each range
        range_pos = jnp.tile(pos[: self.num_agents, :2], (self.num_agents, 1, 1))
        range_pos = jnp.where(
            comm_drop[..., np.newaxis], 0, range_pos
        )  # lost comunicatio data becomes zero
        range_pos = jnp.swapaxes(
            range_pos, 1, 2
        )  # num_agents, 2 (x,y), n_observations (num_agents)
        range_pos_x = jnp.tile(range_pos[:, 0:1], (1, self.num_landmarks, 1))[
            :, :, np.newaxis, :
        ]
        range_pos_y = jnp.tile(range_pos[:, 1:2], (1, self.num_landmarks, 1))[
            :, :, np.newaxis, :
        ]

        # update the tracking buffer
        range_buffer_head = jnp.where(
            range_buffer_head + self.num_agents <= self.tracking_buffer_len,
            range_buffer_head,
            0,
        )
        range_buffer = jax.lax.dynamic_update_slice(
            range_buffer, range_pos_x, (0, 0, 0, range_buffer_head)
        )
        range_buffer = jax.lax.dynamic_update_slice(
            range_buffer, range_pos_y, (0, 0, 1, range_buffer_head)
        )
        range_buffer = jax.lax.dynamic_update_slice(
            range_buffer, land_ranges, (0, 0, 2, range_buffer_head)
        )
        range_buffer_head += self.num_agents

        return range_buffer, range_buffer_head, comm_drop

    @partial(jax.jit, static_argnums=0)
    def ls_predictions(
        self, t: int, range_buffer: chex.Array, pos: chex.Array, ranges: chex.Array
    ) -> chex.Array:
        """Tracking Update minimizing the Least Squares error of the range buffers."""

        # if minimum steps for predicting are not reached, tracking correspond to the actual positions of the agents
        def _dummy_pred(_):
            return jnp.tile(
                pos[: self.num_agents, :3], (1, 1, self.num_landmarks)
            ).reshape(self.num_agents, self.num_landmarks, -1)

        # each agents predicts the positions of the landmarks with the ranges that recived (via measuring and communication)
        def _update(_):
            pos_x = range_buffer[:, :, 0]
            pos_y = range_buffer[:, :, 1]
            pos_xy = range_buffer[:, :, :2]
            z = range_buffer[:, :, 2]
            pred_xy = batched_least_squares(pos_x, pos_y, pos_xy, z)
            pred_z = self.estimate_depth(pred_xy, pos, ranges)
            return jnp.concatenate((pred_xy, pred_z[..., np.newaxis]), axis=-1)

        land_pred_pos = jax.lax.cond(
            t >= self.min_steps_ls,
            true_fun=_update,
            false_fun=_dummy_pred,
            operand=None,
        )

        return land_pred_pos

    @partial(jax.jit, static_argnums=0)
    def pf_predictions(
        self,
        rng: chex.Array,
        pf_state: ParticlesState,
        t: int,
        range_buffer: chex.Array,
        pos: chex.Array,
        ranges: chex.Array,
    ) -> Tuple[ParticlesState, chex.Array]:
        pos_xy = range_buffer[:, :, :2]
        pos_xy = jnp.swapaxes(pos_xy, -1, -2)  # num_agents, num_landmarks, num_obs, xy
        z = range_buffer[:, :, 2]
        mask = (z != 0.0).astype(int)  # range==0 means communication drop

        rng_ = jax.random.split(rng, (self.num_agents, self.num_landmarks))
        pf_state, pred_xy = jax.vmap(jax.vmap(self.pf.step_and_predict))(
            rng_, pf_state, pos_xy, z, mask
        )
        pred_z = self.estimate_depth(pred_xy, pos, ranges)

        return pf_state, jnp.concatenate((pred_xy, pred_z[..., np.newaxis]), axis=-1)

    @partial(jax.jit, static_argnums=0)
    def update_predictions(
        self,
        rng: chex.Array,
        pf_state: ParticlesState,
        t: int,
        range_buffer: chex.Array,
        pos: chex.Array,
        ranges: chex.Array,
    ) -> Tuple[ParticlesState, chex.Array]:

        if self.tracking_method == "ls":
            land_pred_pos = self.ls_predictions(t, range_buffer, pos, ranges)
        elif self.tracking_method == "pf":
            pf_state, land_pred_pos = self.pf_predictions(
                rng, pf_state, t, range_buffer, pos, ranges
            )

        def _dummy_pred(_):
            return jnp.tile(
                pos[: self.num_agents, :3], (1, 1, self.num_landmarks)
            ).reshape(self.num_agents, self.num_landmarks, -1)

        # avoid nans for singular matrix in ls computation
        land_pred_pos = jnp.where(
            jnp.isnan(land_pred_pos), _dummy_pred(None), land_pred_pos
        )

        return pf_state, land_pred_pos

    @partial(jax.jit, static_argnums=0)
    def estimate_depth(
        self, pred_xy: chex.Array, pos: chex.Array, ranges: chex.Array
    ) -> chex.Array:
        # bad depth estimation using pitagora
        pos_xy = pos[: self.num_agents, :2]
        ranges = ranges[:, self.num_agents :]  # ranges between agents and landmarks
        delta_xy = (pred_xy - pos_xy[:, np.newaxis]) ** 2
        to_square = ranges**2 - delta_xy[:, :, 0] - delta_xy[:, :, 1]
        z = pos[: self.num_agents, -1:] + jnp.sqrt(
            jnp.where(to_square > 0, to_square, 0)
        )
        return z

    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: State) -> Dict[str, chex.Array]:

        return {
            agent: jnp.ones((len(self.discrete_actions_mapping),), dtype=jnp.uint8)
            for i, agent in enumerate(self.agents)
        }

    @partial(jax.jit, static_argnums=0)
    def get_init_pos(
        self, rng: chex.Array, min_init_distance: float, max_init_distance: float
    ) -> chex.Array:
        def generate_points(carry, _):
            rng, points, i = carry
            mask = (
                jnp.arange(self.num_entities) >= i
            )  # consider a priori valid the distances with non-done points

            def generate_point(while_state):
                rng, _ = while_state
                rng, _rng = jax.random.split(rng)
                new_point = jax.random.uniform(
                    _rng, (2,), minval=-max_init_distance, maxval=max_init_distance
                )
                return rng, new_point

            def is_valid_point(while_state):
                _, point = while_state
                distances = jnp.sqrt(jnp.sum((points - point) ** 2, axis=-1))
                return ~jnp.all(
                    mask
                    | (
                        (distances >= min_init_distance)
                        & (distances <= max_init_distance)
                    )
                )

            init_point = generate_point((rng, 0))
            rng, new_point = jax.lax.while_loop(
                cond_fun=is_valid_point, body_fun=generate_point, init_val=init_point
            )
            points = points.at[i].set(new_point)
            carry = (rng, points, i + 1)
            return carry, new_point

        rng, _rng = jax.random.split(rng)
        pos = jnp.zeros((self.num_entities, 2))
        pos = pos.at[0].set(
            jax.random.uniform(
                _rng, (2,), minval=-max_init_distance, maxval=max_init_distance
            )
        )  # first point
        (rng, pos, i), _ = jax.lax.scan(
            generate_points, (rng, pos, 1), None, self.num_entities - 1
        )
        return pos
