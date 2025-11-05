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
    # at the moment I haven't found a better way to fill the diagonal with 0s of not-squared matrices
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
    traj_coeffs: (
        chex.Array
    )  # [float]*num_entities, coefficient of linear trajectory models
    traj_intercepts: (
        chex.Array
    )  # [float]*num_entities, intercept of linear trajectory models
    land_pred_pos: (
        chex.Array
    )  # [num_agents, num_landmarks, xyz], current tracking state of each agent for each landmark
    range_buffer: (
        chex.Array
    )  # [num_agents, num_landmarks, (observer_xy, observed_range), len(buffer)], tracking buffer for each agent-landmark pair
    range_buffer_head: int  # head iterator of the tracking buffer
    pf_state: ParticlesState | None  # state of the particle filter
    steps_next_land_action: (
        chex.Array
    )  # [int]*num_landmarks, step until when the landmarks are gonna change directions
    t: int  # step
    cum_rew: float  # cumulative episode reward
    last_actions: chex.Array  # last action taken by the agent
    # Fields for caching range computations between steps_for_new_range intervals
    steps_since_range: int  # steps since last range computation
    cached_delta_xyz: chex.Array  # cached delta_xyz from last range computation
    cached_ranges: chex.Array  # cached ranges from last range computation
    cached_comm_drop: (
        chex.Array
    )  # cached communication drop from last range computation


class UTracking(MultiAgentEnv):

    traj_models = traj_models
    discrete_actions_mapping = jnp.array([-0.24, -0.12, 0, 0.12, 0.24])

    def __init__(
        self,
        num_agents: int = 2,
        num_landmarks: int = 2,
        dt: int = 30,
        max_steps: int = 128,
        discrete_actions: bool = True,
        agent_depth: Tuple[float, float] = (
            0.0,
            0.0003,
        ),  # defines the range of depth for spawning agents
        landmark_depth: Tuple[float, float] = (
            5.0,
            20.0,
        ),  # defines the range of depth for spawning landmarks
        landmark_depth_known: bool = True,  # agents know the depth of the landmarks
        min_valid_distance: float = 5.0,  # under this distance it's considered a crash
        min_init_distance: float = 30.0,  # minimum initial distance between vehicles
        max_init_distance: float = 200.0,  # maximum initial distance between vehicles
        max_range_dist: float = 450.0,  # above this distance can't recieve landmark ranges
        max_comm_dist: float = 1500.0,  # above this distance can't communicate
        prop_agent: int = 30,  # rpm of agent's propellor, defines the speeds for agents (30rpm is ~1m/s)
        landmark_rel_speed: float = (
            0.1,
            0.5,
        ),  # min and maximum relative speed of landmarks to agents
        difficulty: str = "medium",  # difficulty of the environment (manual, easy, medium, hard)
        rudder_range_landmark: Tuple[float, float] = (
            0.10,
            0.25,
        ),  # defines the angle of movement change for landmarks
        dirchange_time_range_landmark: Tuple[int, int] = (
            5,
            15,
        ),  # defines min and max random steps to wait for changing the landmark directions
        tracking_method: str = "pf",  # method for tracking the landmarks positions (ls, pf)
        tracking_buffer_len: int = 32,  # maximum number of range observations kept for predicting the landmark positions
        range_noise_std: float = 10.0,  # standard deviation of the gaussian noise added to range measurements (meters)
        traj_noise_std: float = 0.02,  # standard deviation of the gaussian noise added to the traj models (radians)
        lost_comm_prob=0.1,  # probability of loosing communications (range measurements and intra-agent communication)
        min_steps_ls: int = 2,  # minimum steps for collecting data and start predicting landmarks positions with least squares
        rew_dist_thr: float = 150.0,  # distance threshold for the follow reward
        rew_pred_ideal: float = 10.0,  # ideal prediction error for tracking reward (in meters)
        rew_pred_thr: float = 50.0,  # tracking error threshold for tracking reward
        rew_norm_landmarks: bool = True,  # if true, the reward is normalized by the number of landmarks
        rew_follow_coeff: float = 1.0,  # reward coefficient for following landmarks
        rew_tracking_coeff: float = 1.0,  # reward coefficient for tracking landmarks
        truncate_failed_episode: bool = False,  # if true, the episode is truncated when a crash or lost landmark is detected
        penalty_for_crashing: bool = True,  # if true, the reward is -1 if there is any crash
        penalty_for_lost_agent: bool = True,  # if true, the reward is -1 if any agent loses tracking of any landmark
        pre_init_pos: bool = True,  # computing the initial positions can be expensive if done on the go; to reduce the reset (and therefore step) time, precompute a bunch of possible options
        seed_init_pos: int = 0,  # random seed for precomputing initial distance
        pre_init_pos_len: int = 100000,  # how many initial positions precompute
        ranges_in_obs: bool = False,  # if true, the agents can observed the collected ranges
        matrix_obs: bool = False,  # if true, the obs is a matrix with vertex features relative to all the entities, otherwise flattened
        matrix_state: bool = False,  # if true, the state is represented with vertex features relative to all the entities
        state_as_edges: bool = False,  # if true, the matrix state is represented as edge features, otherwise as vertex features
        space_unit: float = 1e-3,  # unit of space for space observations (default to hundreds of meters)
        infos_for_render: bool = False,  # if true, additional infos are returned for rendering porpouses
        pf_num_particles: int = 5000,  # number of particles for the particle filter
        actions_as_angles: bool = False,  # if true, the actions are interpreted as angles
        normalize_distances: bool = True,  # if true, distances in observations and states are normalized to [0,1] based on max_range_dist
        steps_for_new_range: int = 1,  # number of steps between new range observations
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
        assert tracking_method in [
            "ls",
            "pf",
        ], "tracking method must be ls (Least Squares) or pf (Particle Filter)"
        assert difficulty in [
            "manual",
            "easy",
            "medium",
            "hard",
            "expert",
        ], "difficulty must be manual, easy, medium, hard or expert"

        # LS method only works with steps_for_new_range=1 to avoid complexity
        assert (
            tracking_method != "ls" or steps_for_new_range == 1
        ), "LS tracking method requires steps_for_new_range=1"

        if difficulty != "manual":
            if difficulty == "easy":
                landmark_rel_speed = (0.0, 0.35)
                dirchange_time_range_landmark = (5, 15)
            elif difficulty == "medium":
                landmark_rel_speed = (0.15, 0.5)
                dirchange_time_range_landmark = (2, 10)
            elif difficulty == "hard":
                landmark_rel_speed = (0.5, 0.7)
                dirchange_time_range_landmark = (5, 15)
            elif difficulty == "expert":
                landmark_rel_speed = (0.83, 0.86)
                dirchange_time_range_landmark = (5, 15)

        self.max_steps = max_steps
        self.num_agents = num_agents
        self.num_landmarks = num_landmarks
        self.num_entities = num_agents + num_landmarks
        self.agents = [f"agent_{i}" for i in range(1, num_agents + 1)]
        self.landmarks = [f"landmark_{i}" for i in range(1, num_landmarks + 1)]
        self.entities = self.agents + self.landmarks
        self.discrete_actions = discrete_actions
        self.actions_as_angles = actions_as_angles
        self.actions_as_angles = actions_as_angles
        self.agent_depth = agent_depth
        self.landmark_rel_speed = landmark_rel_speed
        self.landmark_depth = landmark_depth
        self.landmark_depth_known = landmark_depth_known
        self.landmark_depth_known = landmark_depth_known
        self.min_valid_distance = min_valid_distance
        self.min_init_distance = min_init_distance
        self.max_init_distance = max_init_distance
        self.max_range_dist = max_range_dist
        self.max_comm_dist = max_comm_dist
        self.max_comm_dist = max_comm_dist
        self.prop_agent = prop_agent
        self.rudder_range_landmark = np.array(rudder_range_landmark)
        self.dirchange_time_range_landmark = dirchange_time_range_landmark
        self.tracking_method = tracking_method
        self.tracking_buffer_len = tracking_buffer_len
        self.range_noise_std = range_noise_std
        self.traj_noise_std = traj_noise_std
        self.traj_noise_std = traj_noise_std
        self.lost_comm_prob = lost_comm_prob
        self.min_steps_ls = min_steps_ls
        self.rew_dist_thr = rew_dist_thr
        self.rew_pred_ideal = rew_pred_ideal
        self.rew_pred_thr = rew_pred_thr
        self.rew_norm_landmarks = rew_norm_landmarks
        self.rew_follow_coeff = rew_follow_coeff
        self.rew_tracking_coeff = rew_tracking_coeff
        self.penalty_for_crashing = penalty_for_crashing
        self.penalty_for_lost_agent = penalty_for_lost_agent
        self.truncate_failed_episode = truncate_failed_episode
        self.pre_init_pos = pre_init_pos
        self.pre_init_pos_len = pre_init_pos_len
        self.ranges_in_obs = ranges_in_obs
        self.matrix_obs = matrix_obs
        self.matrix_state = matrix_state
        self.state_as_edges = state_as_edges
        self.space_unit = space_unit
        self.infos_for_render = infos_for_render
        self.steps_for_new_range = steps_for_new_range

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
        self.obs_feats = 6  # feat for each entity, [obs_delta_x, obs_delta_y, obs_delta_z, range, is_agent, is_self]
        if self.matrix_obs:
            self.observation_spaces = {
                i: Box(-jnp.inf, jnp.inf, (self.num_entities, self.obs_feats))
                for i in self.agents
            }
        else:
            self.observation_spaces = {
                i: Box(-jnp.inf, jnp.inf, (self.num_entities * self.obs_feats,))
                for i in self.agents
            }

        # world state shape
        # if edges: [delta_x, delta_y, delta_z, obs_delta_x, obs_delta_y, obs_delta_z, range, is_self(oh)]
        # else: [pos_x, pos_y, pos_z, direction, vel, pred_error, is_agent]
        self.state_features = 9 if self.state_as_edges else 7
        if self.matrix_state:
            self.world_state_space = Box(
                -jnp.inf, jnp.inf, (self.num_entities, self.state_features)
            )
        else:
            self.world_state_space = Box(
                -jnp.inf, jnp.inf, (self.num_entities * self.state_features,)
            )

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
        valid_prop = (
            lambda p: p >= self.prop_agent * self.landmark_rel_speed[0]
            and p <= self.prop_agent * self.landmark_rel_speed[1]
        )
        self.idx_valid_traj_model_landmarks = jnp.array(
            [i for i, p in enumerate(self.traj_model_prop) if valid_prop(p)]
        )

        if normalize_distances:
            self.normalize_distances = lambda x: jnp.clip(
                x / self.max_range_dist, 0.0, 1.0
            )
        else:
            self.normalize_distances = lambda x: x

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
        return f"utracking_{self.num_agents}v{self.num_landmarks}"

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
            key_comm, ranges_real_2d, ranges_2d, pos, range_buffer, range_buffer_head
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
        obs = self.get_obs(delta_xyz, ranges, comm_drop, pos, pos, land_pred_pos)
        obs["world_state"] = self.get_global_state(
            delta_xyz, ranges, comm_drop, pos, vel, land_pred_pos
        )

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
            last_actions=jnp.zeros(self.num_agents),
            steps_since_range=0,
            cached_delta_xyz=delta_xyz,
            cached_ranges=ranges,
            cached_comm_drop=comm_drop,
        )
        return obs, state

    @partial(jax.jit, static_argnums=0)
    def world_step(
        self,
        rng: chex.PRNGKey,
        actions: chex.Array,
        pos: chex.Array,
        vel: chex.Array,
        traj_coeffs: chex.Array,
        traj_intercepts: chex.Array,
    ) -> chex.Array:

        if self.actions_as_angles:
            pos = pos.at[:, -1].set(actions)
        else:
            # update the angle
            angle_change = actions * traj_coeffs + traj_intercepts
            # add noise
            angle_change += (
                jax.random.normal(rng, shape=angle_change.shape) * self.traj_noise_std
            )
            new_angles = (pos[:, -1] + angle_change + jnp.pi) % (2 * jnp.pi) - jnp.pi
            # update the x-y position (depth remains constant)
            pos = pos.at[:, -1].set(new_angles)

        if self.actions_as_angles:
            pos = pos.at[:, -1].set(actions)
        else:
            # update the angle
            angle_change = actions * traj_coeffs + traj_intercepts
            # add noise
            angle_change += (
                jax.random.normal(rng, shape=angle_change.shape) * self.traj_noise_std
            )
            new_angles = (pos[:, -1] + angle_change + jnp.pi) % (2 * jnp.pi) - jnp.pi
            # update the x-y position (depth remains constant)
            pos = pos.at[:, -1].set(new_angles)

        pos = pos.at[:, 0].add(jnp.cos(pos[:, -1]) * vel * self.dt)
        pos = pos.at[:, 1].add(jnp.sin(pos[:, -1]) * vel * self.dt)
        return pos

    @partial(jax.jit, static_argnums=0)
    def step_env(
        self,
        rng: chex.PRNGKey,
        state: State,
        actions: dict,
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool]]:

        # preprocess actions
        agent_actions = jnp.array([actions[a] for a in self.agents])
        last_actions = agent_actions.astype(float)
        agent_actions = self.preprocess_actions(agent_actions)
        landmark_actions, steps_next_land_action = self.get_landmarks_actions(
            rng, state.steps_next_land_action, state.t
        )

        if (
            self.actions_as_angles
        ):  # this is used to inject angle actions in the environment (mainly for debugging)
            actions = jnp.array([actions[a] for a in self.entities])
        else:
            actions = jnp.concatenate((agent_actions, landmark_actions))

        rng, _rng = jax.random.split(rng)
        pos = self.world_step(
            _rng,
            actions,
            state.pos,
            state.vel,
            state.traj_coeffs,
            state.traj_intercepts,
        )

        # update tracking - conditionally compute new ranges based on steps_for_new_range
        steps_since_range = state.steps_since_range + 1
        should_compute_ranges = steps_since_range >= self.steps_for_new_range

        def compute_new_ranges():
            rng_, key_ranges, key_comm = jax.random.split(rng, 3)
            delta_xyz_, ranges_real_2d_, ranges_real_, ranges_2d_, ranges_ = (
                self.get_ranges(key_ranges, pos)
            )
            range_buffer_, range_buffer_head_, comm_drop_ = self.communicate(
                key_comm,
                ranges_real_2d_,
                ranges_2d_,
                pos,
                state.range_buffer,
                state.range_buffer_head,
            )
            rng_pred, rng_rest = jax.random.split(rng_)
            pf_state_, land_pred_pos_ = self.update_predictions(
                rng_pred, state.pf_state, state.t, range_buffer_, pos, ranges_2d_
            )
            return (
                delta_xyz_,
                ranges_real_2d_,
                ranges_,
                comm_drop_,
                range_buffer_,
                range_buffer_head_,
                pf_state_,
                land_pred_pos_,
                0,
                rng_rest,
            )

        def use_cached_ranges():
            # Use cached values and previous predictions, but compute actual ranges for reward
            # (since rewards depend on actual distances)
            rng_, _ = jax.random.split(rng)
            _, ranges_real_2d_current, _, _, _ = self.get_ranges(rng_, pos)
            return (
                state.cached_delta_xyz,
                ranges_real_2d_current,
                state.cached_ranges,
                state.cached_comm_drop,
                state.range_buffer,
                state.range_buffer_head,
                state.pf_state,
                state.land_pred_pos,
                steps_since_range,
                rng,
            )

        (
            delta_xyz,
            ranges_real_2d,
            ranges,
            comm_drop,
            range_buffer,
            range_buffer_head,
            pf_state,
            land_pred_pos,
            new_steps_since_range,
            rng_after,
        ) = jax.lax.cond(should_compute_ranges, compute_new_ranges, use_cached_ranges)

        # get global reward, done, info
        reward, done, info = self.get_rew_done_info(
            state.t,
            pos,
            ranges,
            ranges_real_2d,
            land_pred_pos,
            state.cum_rew,
        )
        rewards = {agent: reward for agent in self.agents}
        done = {agent: done for agent in self.agents + ["__all__"]}

        # agents obs and global state
        obs = self.get_obs(delta_xyz, ranges, comm_drop, pos, state.pos, land_pred_pos)
        obs["world_state"] = self.get_global_state(
            delta_xyz, ranges, comm_drop, pos, state.vel, land_pred_pos
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
            last_actions=last_actions,
            steps_since_range=new_steps_since_range,
            cached_delta_xyz=jax.lax.cond(
                should_compute_ranges, lambda: delta_xyz, lambda: state.cached_delta_xyz
            ),
            cached_ranges=jax.lax.cond(
                should_compute_ranges, lambda: ranges, lambda: state.cached_ranges
            ),
            cached_comm_drop=jax.lax.cond(
                should_compute_ranges, lambda: comm_drop, lambda: state.cached_comm_drop
            ),
        )
        return obs, state, rewards, done, info

    @partial(jax.jit, static_argnums=0)
    def get_obs(self, delta_xyz, ranges, comm_drop, pos, old_pos, land_pred_pos):
        # first a matrix with all the observations is created, composed by
        # the position of the agent or the relative position of other agents (comunication) and landmarks (tracking)
        # the absolute distance (ranges) is_agent, is_self features
        # [pos_x, pos_y, pos_z, dist, is_agent, is_self]*n_entities

        delta_self_pos = pos - old_pos
        delta_self_pos = delta_self_pos.at[:, :3].set(
            self.normalize_distances(delta_self_pos[:, :3])
        )
        # normalize angle difference to [0, 1]
        delta_self_pos = delta_self_pos.at[:, 3].set(
            (delta_self_pos[:, 3] + jnp.pi) / (2 * jnp.pi)
        )
        delta_xyz = self.normalize_distances(delta_xyz)

        other_agents_dist = jnp.where(
            comm_drop[:, :, None], 0, delta_xyz[:, : self.num_agents]
        )  # 0 for communication drop
        self_mask = (
            jnp.arange(self.num_agents) == np.arange(self.num_agents)[:, np.newaxis]
        )
        self_pos_feats = delta_self_pos[: self.num_agents, [0, 1, 3]]
        self_pos_feats = self_pos_feats.at[:, 2].set(0)
        agents_rel_pos = jnp.where(
            self_mask[:, :, None],
            self_pos_feats,
            other_agents_dist,
        )  # for self use delta with respect to previous position
        lands_rel_pos = (
            pos[: self.num_agents, None, :3] - land_pred_pos
        )  # relative distance from predicted positions
        lands_rel_pos = self.normalize_distances(lands_rel_pos)
        pos_feats = jnp.concatenate((agents_rel_pos, lands_rel_pos), axis=1)
        is_agent_feat = jnp.tile(
            jnp.concatenate((jnp.ones(self.num_agents), jnp.zeros(self.num_landmarks))),
            (self.num_agents, 1),
        )
        is_self_feat = (
            jnp.arange(self.num_entities) == jnp.arange(self.num_agents)[:, np.newaxis]
        )
        ranges *= 1.0 if self.ranges_in_obs else 0.0  # mask the ranges if not in obs
        # the distance based feats are rescaled to hundreds of meters (better for NNs)

        feats = jnp.concatenate(
            (
                pos_feats,
                self.normalize_distances(ranges[:, :, None]),
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
    def get_vertex_state(
        self,
        pos: chex.Array,
        vel: chex.Array,
        ranges: chex.Array,
        land_pred_pos: chex.Array,
    ) -> chex.Array:
        # state as vertex features
        # state is obs, vel, is_agent, pred_error for each entity

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
            (
                pos * self.space_unit,
                vel[:, None],
                preds[:, None] * self.space_unit,
                is_agent[:, None],
            ),
            axis=-1,
        )
        return state

    @partial(jax.jit, static_argnums=0)
    def get_edges_state(
        self,
        delta_xyz: chex.Array,
        ranges: chex.Array,
        comm_drop: chex.Array,
        pos: chex.Array,
        land_pred_pos: chex.Array,
    ) -> chex.Array:

        def agent_edge_feats(aidx, delta_xyz_, ranges_, comm_drop_, land_pred_pos_):
            """computes the edge features for the agent aidx"""
            others_delta = jnp.delete(
                delta_xyz_, aidx, axis=0, assume_unique_indices=True
            )  # real distance in xyz

            # observed distance other agents (xyz)
            other_agents_dist = jnp.where(
                comm_drop_[:, None], 0, delta_xyz_[: self.num_agents]
            )
            other_agents_dist = jnp.delete(
                other_agents_dist, aidx, axis=0, assume_unique_indices=True
            )  # delete self

            # observed distance other agents+landmarks (xyz)
            others_observed_delta = jnp.vstack((other_agents_dist, land_pred_pos_))

            # the other is_agent
            is_agent = jnp.vstack(
                (
                    jnp.zeros((self.num_agents - 1, 2)).at[:, 0].set(1),  # is agent
                    jnp.zeros((self.num_landmarks, 2)).at[:, 1].set(1),  # is landmark
                )
            )

            others_ranges = jnp.delete(ranges_, aidx, assume_unique_indices=True)
            edges_feats = jnp.concatenate(
                (
                    others_delta,
                    others_observed_delta,
                    others_ranges[:, None],  # delete self
                    is_agent,
                ),
                axis=1,
            )
            return edges_feats

        edges = jax.vmap(agent_edge_feats)(
            jnp.arange(self.num_agents),
            delta_xyz * self.space_unit,
            ranges * self.space_unit,
            comm_drop,
            land_pred_pos * self.space_unit,
        )

        return jnp.vstack(edges)

    @partial(jax.jit, static_argnums=0)
    def get_global_state(
        self,
        delta_xyz: chex.Array,
        ranges: chex.Array,
        comm_drop: chex.Array,
        pos: chex.Array,
        vel: chex.Array,
        land_pred_pos: chex.Array,
    ):

        if self.state_as_edges:
            state = self.get_edges_state(
                delta_xyz, ranges, comm_drop, pos, land_pred_pos
            )
        else:
            state = self.get_vertex_state(pos, vel, ranges, land_pred_pos)

        if self.matrix_state:
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

        # REWARDS
        # reward based on number of landmarks followed, i.e. that have at least one agent at min distance

        follow_rew = (min_land_dist <= self.rew_dist_thr).sum()

        # reward based on the tracking error
        # exponential decay of the reward based on the tracking error
        # goes to 0 if the tracking error is above the threshold
        # goes to 1 if the tracking error is 0
        def exponential_decay(x, x1=self.rew_pred_ideal, x2=self.rew_pred_thr):
            t = (x - x1) / (x2 - x1)
            t = jnp.clip(t, 0, 1 - 1e-10)  # Avoid division by zero
            return jnp.where(x < x1, 1, jnp.where(x > x2, 0, jnp.exp(-2 * t / (1 - t))))

        tracking_rew = (jax.vmap(exponential_decay)(pred_2d_err)).sum()

        rew = (
            self.rew_follow_coeff * follow_rew + self.rew_tracking_coeff * tracking_rew
        )

        if self.rew_norm_landmarks:
            rew /= self.num_landmarks

        # penalize crashing between agents
        if self.penalty_for_crashing:
            rew = jnp.where((agent_dist < self.min_valid_distance).any(), -1.0, rew)

        # penalize if agent lost all landmarks
        # i.e. any agent is at distance > max_range_dist*2 from all landmarks
        if self.penalty_for_lost_agent:
            any_agent_lost = (
                distances_2d[:, self.num_agents :].max(axis=1)
                >= self.max_range_dist * 2
            ).any()
            rew = jnp.where(any_agent_lost, -1.0, rew)

        # DONE
        done = t == self.max_steps

        # truncate the episode if failed (if a landmark is lost)
        if self.truncate_failed_episode:
            failed_episode = (
                min_land_dist >= self.max_range_dist * 2
            ).any()  # failed episode if a landmark is lost
            done = done | failed_episode

        # INFO
        # return different infos if the env is gonna be used for rendering or training
        info = {
            "follow_rew": follow_rew,
            "tracking_rew": tracking_rew,
            "landmarks_covered": (min_land_dist <= self.rew_dist_thr).sum(keepdims=True)
            / self.num_landmarks,
            "landmarks_lost": (min_land_dist >= self.max_range_dist).sum(keepdims=True)
            / self.num_landmarks,
            "agents_lost": (
                distances_2d[:, self.num_agents :].max(axis=1)
                >= self.max_range_dist * 2
            ).sum(keepdims=True)
            / self.num_agents,
            "tracking_error_mean": pred_2d_err.mean(keepdims=True),
            "land_dist_mean": min_land_dist.mean(keepdims=True),
            "crash": (agent_dist < self.min_valid_distance).any(keepdims=True),
            "normalized_reward": cum_rew + rew / self.max_steps,
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
        if self.actions_as_angles:
            return actions
        if self.actions_as_angles:
            return actions
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

        ranges = ranges_real + noise
        lost_range = (
            jax.random.uniform(key_lost, shape=ranges.shape) <= self.lost_comm_prob
        ) | (
            ranges_real > self.max_range_dist
        )  # lost communication or landmark too far
        ranges = jnp.where(lost_range, 0.0, ranges)
        lost_range = (
            jax.random.uniform(key_lost, shape=ranges.shape) <= self.lost_comm_prob
        ) | (
            ranges_real > self.max_range_dist
        )  # lost communication or landmark too far
        ranges = jnp.where(lost_range, 0.0, ranges)
        ranges = fill_diagonal_zeros(ranges)  # reset to 0s the self-ranges

        ranges_2d = ranges_real_2d + noise
        ranges_2d = jnp.where(lost_range, 0.0, ranges_2d)
        ranges_2d = fill_diagonal_zeros(ranges_2d)

        ranges_2d = ranges_real_2d + noise
        ranges_2d = jnp.where(lost_range, 0.0, ranges_2d)
        ranges_2d = fill_diagonal_zeros(ranges_2d)

        return delta_xyz, ranges_real_2d, ranges_real, ranges_2d, ranges

    @partial(jax.jit, static_argnums=0)
    def communicate(
        self,
        rng: chex.Array,
        ranges_real: chex.Array,
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

        # communication is dropped also for the agents that are too far
        comm_drop = comm_drop | (
            ranges_real[:, : self.num_agents] > self.max_comm_dist
        )  # too far

        # communication is dropped also for the agents that are too far
        comm_drop = comm_drop | (
            ranges_real[:, : self.num_agents] > self.max_comm_dist
        )  # too far
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

        if self.landmark_depth_known:
            # repeat the depth of the landmarks for each agent
            return jnp.tile(pos[self.num_agents :, 2], (self.num_agents)).reshape(
                self.num_agents, self.num_landmarks
            )

        if self.landmark_depth_known:
            # repeat the depth of the landmarks for each agent
            return jnp.tile(pos[self.num_agents :, 2], (self.num_agents)).reshape(
                self.num_agents, self.num_landmarks
            )

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

        if not self.discrete_actions:
            raise NotImplementedError(
                '"get_avail_actions" supports only discrete actions'
            )

        def close_action_valid(last_action_idx):
            avail_actions = jnp.zeros(len(self.discrete_actions_mapping))
            avail_actions = avail_actions.at[last_action_idx].set(1)
            return jnp.where(
                last_action_idx == 0,
                avail_actions.at[1].set(1),
                jnp.where(
                    last_action_idx == len(self.discrete_actions_mapping) - 1,
                    avail_actions.at[-2].set(1),
                    jnp.minimum(
                        avail_actions.at[last_action_idx - 1].set(1)
                        + avail_actions.at[last_action_idx + 1].set(1),
                        1,
                    ),
                ),
            )

        avail_actions = jax.vmap(close_action_valid)(state.last_actions.astype(int))
        return {a: avail_actions[i] for i, a in enumerate(self.agents)}

        if not self.discrete_actions:
            raise NotImplementedError(
                '"get_avail_actions" supports only discrete actions'
            )

        def close_action_valid(last_action_idx):
            avail_actions = jnp.zeros(len(self.discrete_actions_mapping))
            avail_actions = avail_actions.at[last_action_idx].set(1)
            return jnp.where(
                last_action_idx == 0,
                avail_actions.at[1].set(1),
                jnp.where(
                    last_action_idx == len(self.discrete_actions_mapping) - 1,
                    avail_actions.at[-2].set(1),
                    jnp.minimum(
                        avail_actions.at[last_action_idx - 1].set(1)
                        + avail_actions.at[last_action_idx + 1].set(1),
                        1,
                    ),
                ),
            )

        avail_actions = jax.vmap(close_action_valid)(state.last_actions.astype(int))
        return {a: avail_actions[i] for i, a in enumerate(self.agents)}

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
