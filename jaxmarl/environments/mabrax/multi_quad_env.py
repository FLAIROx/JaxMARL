# multi_quad_env.py

"""
A Brax/MJX version of a multi-rotor quadcopter team with payload.
This environment adapts the original MuJoCo-based implementation to Brax using the MJX backend.
"""

from brax import base, envs
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
import jax
from jax import numpy as jp
import mujoco
import os

# Helper functions in JAX (converted from numpy/numba)
def jp_R_from_quat(q: jp.ndarray) -> jp.ndarray:
  """Compute rotation matrix from quaternion [w, x, y, z]."""
  q = q / jp.linalg.norm(q)
  w, x, y, z = q[0], q[1], q[2], q[3]
  r1 = jp.array([1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)])
  r2 = jp.array([2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)])
  r3 = jp.array([2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)])
  return jp.stack([r1, r2, r3])

def jp_angle_between(v1: jp.ndarray, v2: jp.ndarray) -> jp.ndarray:
  """Computes the angle between two vectors."""
  norm1 = jp.linalg.norm(v1)
  norm2 = jp.linalg.norm(v2)
  dot = jp.dot(v1, v2)
  cos_theta = dot / (norm1 * norm2 + 1e-6)
  cos_theta = jp.clip(cos_theta, -1.0, 1.0)
  return jp.arccos(cos_theta)


class MultiQuadEnv(PipelineEnv):
  """
  A Brax/MJX environment for a multi-rotor quadcopter team carrying a payload.

  The environment is initialized from a MuJoCo XML model and then converted into a Brax
  system using the MJX backend. The control actions (in [-1, 1]) are scaled into thrust commands.
  """
  def __init__(
      self,
      policy_freq: float = 250,              # Policy frequency in Hz.
      sim_steps_per_action: int = 1,           # Physics steps between control actions.
      max_time: float = 10.0,                  # Maximum simulation time per episode.
      reset_noise_scale: float = 0.1,          # Noise scale for initial state reset.
      **kwargs,
  ):
    print("Initializing MultiQuadEnv")
    # Load the MJX model from the XML file.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(current_dir, "mujoco", "two_quad_payload.xml")
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    # Convert the MuJoCo model to a Brax system.
    sys = mjcf.load_model(mj_model)
    kwargs['n_frames'] = kwargs.get('n_frames', sim_steps_per_action)
    kwargs['backend'] = 'mjx'
    super().__init__(sys, **kwargs)

    # Save environment parameters.
    self.policy_freq = policy_freq
    self.sim_steps_per_action = sim_steps_per_action
    self.time_per_action = 1.0 / self.policy_freq
    self.max_time = max_time
    self._reset_noise_scale = reset_noise_scale
    self.warmup_time = 1.0

    # Set simulation timestep based on policy frequency and steps per action.
    dt = self.time_per_action / self.sim_steps_per_action
    sys.mj_model.opt.timestep = dt

    # Maximum thrust from the original environment.
    self.max_thrust = 0.11772
    # Define the target goal for the payload.
    self.goal_center = jp.array([0.0, 0.0, 1.0])
    self.goal_radius = 0.8  # sphere radius for random goal position
    self.target_position = self.goal_center

    # Cache body IDs for fast lookup.
    self.payload_body_id = mujoco.mj_name2id(
        sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, "payload")
    self.q1_body_id = mujoco.mj_name2id(
        sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, "q0_container")
    self.q2_body_id = mujoco.mj_name2id(
        sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, "q1_container")
    
    # Cache joint IDs for fast lookup.
    self.payload_joint_id = mujoco.mj_name2id(
        sys.mj_model, mujoco.mjtObj.mjOBJ_JOINT.value, "payload")
    self.q1_joint_id = mujoco.mj_name2id(
        sys.mj_model, mujoco.mjtObj.mjOBJ_JOINT.value, "q0_container")
    self.q2_joint_id = mujoco.mj_name2id(
        sys.mj_model, mujoco.mjtObj.mjOBJ_JOINT.value, "q1_container")

    # Cache the starting indices in qpos using the joint qpos addresses.
    self.payload_qpos_start = sys.mj_model.jnt_qposadr[self.payload_joint_id]
    self.q1_qpos_start = sys.mj_model.jnt_qposadr[self.q1_joint_id]
    self.q2_qpos_start = sys.mj_model.jnt_qposadr[self.q2_joint_id]

    print("IDs:")
    print("Payload body ID:", self.payload_body_id)
    print("Quad 1 body ID:", self.q1_body_id)
    print("Quad 2 body ID:", self.q2_body_id)
    print("Payload joint ID:", self.payload_joint_id)
    print("Quad 1 joint ID:", self.q1_joint_id)
    print("Quad 2 joint ID:", self.q2_joint_id)
    print("Payload qpos start:", self.payload_qpos_start)
    print("Quad 1 qpos start:", self.q1_qpos_start)
    print("Quad 2 qpos start:", self.q2_qpos_start)
    print("MultiQuadEnv initialized successfully.")

  @staticmethod
  def generate_configuration(key):
    """
    Generate a candidate configuration from random samples.
    Returns:
      payload_pos, quad1_pos, quad2_pos -- each a jp.array of shape (3,)
    """
    subkeys = jax.random.split(key, 9)
    min_quad_z = 0.008    # quad on ground
    min_payload_z = 0.0055  # payload on ground

    # Payload position: x,y uniformly in [-1.5, 1.5] and z in [-1.0, 3.0] then clipped to [0, 3]
    payload_xy = jax.random.uniform(subkeys[0], (2,), minval=-1.5, maxval=1.5)
    payload_z = jax.random.uniform(subkeys[1], (), minval=-1.0, maxval=3.0)
    payload_pos = jp.array([payload_xy[0], payload_xy[1], payload_z])

    # Parameters for Quad positions.
    mean_r   = 0.3
    std_r    = 0.1
    clip_min = 0.05
    clip_max = 0.3
    mean_theta  = jp.pi / 7
    mean_theta2 = -jp.pi / 7
    std_theta   = jp.pi / 8
    std_phi     = jp.pi / 3

    # Quad 1.
    r1     = jp.clip(mean_r + std_r * jax.random.normal(subkeys[2], ()), clip_min, clip_max)
    theta1 = mean_theta + std_theta * jax.random.normal(subkeys[4], ())
    # Quad 2.
    r2     = jp.clip(mean_r + std_r * jax.random.normal(subkeys[3], ()), clip_min, clip_max)
    theta2 = mean_theta2 + std_theta * jax.random.normal(subkeys[5], ())

    # Common phi offset and individual noise.
    phi_offset = jax.random.uniform(subkeys[6], (), minval=-jp.pi, maxval=jp.pi)
    phi1 = std_phi * jax.random.normal(subkeys[7], ()) + phi_offset
    phi2 = std_phi * jax.random.normal(subkeys[8], ()) + phi_offset

    # Convert spherical to Cartesian for Quad 1.
    quad1_x = r1 * jp.sin(theta1) * jp.cos(phi1) + payload_pos[0]
    quad1_y = r1 * jp.sin(theta1) * jp.sin(phi1) + payload_pos[1]
    quad1_z = jp.clip(r1 * jp.cos(theta1) + payload_pos[2], min_quad_z, 3)
    quad1_pos = jp.array([quad1_x, quad1_y, quad1_z])
    
    # Convert spherical to Cartesian for Quad 2.
    quad2_x = r2 * jp.sin(theta2) * jp.cos(phi2) + payload_pos[0]
    quad2_y = r2 * jp.sin(theta2) * jp.sin(phi2) + payload_pos[1]
    quad2_z = jp.clip(r2 * jp.cos(theta2) + payload_pos[2], min_quad_z, 3)
    quad2_pos = jp.array([quad2_x, quad2_y, quad2_z])

    # Ensure payload is above ground.
    payload_pos_z = jp.clip(payload_z, min_payload_z, 3)
    payload_pos = jp.array([payload_xy[0], payload_xy[1], payload_pos_z])
    
    return payload_pos, quad1_pos, quad2_pos

  @staticmethod
  def generate_valid_configuration(key, oversample=3):
    """
    Generate a single valid configuration.
    Oversample candidates and select the first one that meets:
      - Distance between quads >= 0.12.
      - Each quad is at least 0.06 away from the payload.
    """
    candidate_keys = jax.random.split(key, oversample)
    candidate_payload, candidate_quad1, candidate_quad2 = jax.vmap(MultiQuadEnv.generate_configuration)(candidate_keys)
    
    dist_quads = jp.linalg.norm(candidate_quad1 - candidate_quad2, axis=1)
    dist_q1_payload = jp.linalg.norm(candidate_quad1 - candidate_payload, axis=1)
    dist_q2_payload = jp.linalg.norm(candidate_quad2 - candidate_payload, axis=1)
    
    valid_mask = (dist_quads >= 0.12) & (dist_q1_payload >= 0.06) & (dist_q2_payload >= 0.06)
    valid_index = jp.argmax(valid_mask)  # returns 0 if none are valid
    
    return candidate_payload[valid_index], candidate_quad1[valid_index], candidate_quad2[valid_index]

  def reset(self, rng: jax.Array) -> State:
    """Resets the environment to an initial state."""
    rng, rng1, rng2, rng_config = jax.random.split(rng, 4)
    base_qpos = self.sys.qpos0  # Start with the reference configuration.
    qvel = jax.random.uniform(
        rng2, (self.sys.nv,),
        minval=-self._reset_noise_scale,
        maxval=self._reset_noise_scale
    )

    # Get new positions for payload and both quadrotors.
    payload_pos, quad1_pos, quad2_pos = MultiQuadEnv.generate_valid_configuration(rng_config)

    # Generate new orientations (as quaternions) for the quadrotors.
    rng, rng_euler = jax.random.split(rng, 2)
    keys = jax.random.split(rng_euler, 6)
    std_dev = 10 * jp.pi / 180   # 10° in radians.
    clip_val = 60 * jp.pi / 180  # 60° in radians.

    # Quadrotor 1: sample roll and pitch (clipped) and yaw uniformly.
    roll_q1 = jp.clip(jax.random.normal(keys[0]) * std_dev, -clip_val, clip_val)
    pitch_q1 = jp.clip(jax.random.normal(keys[1]) * std_dev, -clip_val, clip_val)
    yaw_q1 = jax.random.uniform(keys[2], minval=-jp.pi, maxval=jp.pi)
    # Quadrotor 2: similarly.
    roll_q2 = jp.clip(jax.random.normal(keys[3]) * std_dev, -clip_val, clip_val)
    pitch_q2 = jp.clip(jax.random.normal(keys[4]) * std_dev, -clip_val, clip_val)
    yaw_q2 = jax.random.uniform(keys[5], minval=-jp.pi, maxval=jp.pi)

    def euler_to_quat(roll, pitch, yaw):
        cr = jp.cos(roll * 0.5)
        sr = jp.sin(roll * 0.5)
        cp = jp.cos(pitch * 0.5)
        sp = jp.sin(pitch * 0.5)
        cy = jp.cos(yaw * 0.5)
        sy = jp.sin(yaw * 0.5)
        # MJX uses [x, y, z, w] order.
        return jp.array([
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ])

    quat_q1 = euler_to_quat(roll_q1, pitch_q1, yaw_q1)
    quat_q2 = euler_to_quat(roll_q2, pitch_q2, yaw_q2)

    # Build the full qpos vector.
    new_qpos = base_qpos
    # Update payload position.
    new_qpos = new_qpos.at[self.payload_qpos_start:self.payload_qpos_start+3].set(payload_pos)
    # Update quadrotor positions.
    new_qpos = new_qpos.at[self.q1_qpos_start:self.q1_qpos_start+3].set(quad1_pos)
    new_qpos = new_qpos.at[self.q2_qpos_start:self.q2_qpos_start+3].set(quad2_pos)
    # Update quadrotor orientations (starting 3 elements later).
    new_qpos = new_qpos.at[self.q1_qpos_start+3:self.q1_qpos_start+7].set(quat_q1)
    new_qpos = new_qpos.at[self.q2_qpos_start+3:self.q2_qpos_start+7].set(quat_q2)

    # print new_qpos
    jax.debug.print("new_qpos: {}", new_qpos)
    
    
    pipeline_state = self.pipeline_init(new_qpos, qvel)
    last_action = jp.zeros(self.sys.nu)
    obs = self._get_obs(pipeline_state, last_action, self.target_position)
    reward = jp.array(0.0)
    done = jp.array(0.0)
    metrics = {'time': pipeline_state.time, 'reward': reward}
    return State(pipeline_state, obs, reward, done, metrics)
  def step(self, state: State, action: jax.Array) -> State:
    """Advances the environment by one control step."""
    # Extract previous action from the observation.
    prev_last_action = state.obs[-(self.sys.nu+1):-1]
    # Scale actions from [-1, 1] to thrust commands in [0, max_thrust].
    thrust_cmds = 0.5 * (action + 1.0)
    action_scaled = thrust_cmds * self.max_thrust

    data0 = state.pipeline_state
    pipeline_state = self.pipeline_step(data0, action_scaled)

    # Compute orientation and collision/out-of-bound checks.
    q1_orientation = pipeline_state.xquat[self.q1_body_id]
    q2_orientation = pipeline_state.xquat[self.q2_body_id]
    up = jp.array([0.0, 0.0, 1.0])
    q1_local_up = jp_R_from_quat(q1_orientation)[:, 2]
    q2_local_up = jp_R_from_quat(q2_orientation)[:, 2]
    angle_q1 = jp_angle_between(q1_local_up, up)
    angle_q2 = jp_angle_between(q2_local_up, up)

    quad1_pos = pipeline_state.xpos[self.q1_body_id]
    quad2_pos = pipeline_state.xpos[self.q2_body_id]
    quad_distance = jp.linalg.norm(quad1_pos - quad2_pos)
    collision = quad_distance < 0.11
    out_of_bounds = jp.logical_or(jp.absolute(angle_q1) > jp.radians(80),
                                  jp.absolute(angle_q2) > jp.radians(80))
    out_of_bounds = jp.logical_or(out_of_bounds, pipeline_state.xpos[self.q1_body_id][2] < 0.05)
    out_of_bounds = jp.logical_or(out_of_bounds, pipeline_state.xpos[self.q2_body_id][2] < 0.05)
    out_of_bounds = jp.logical_or(out_of_bounds, pipeline_state.xpos[self.q1_body_id][2] < pipeline_state.xpos[self.payload_body_id][2])
    out_of_bounds = jp.logical_or(out_of_bounds, pipeline_state.xpos[self.q2_body_id][2] < pipeline_state.xpos[self.payload_body_id][2])
    out_of_bounds = jp.logical_or(out_of_bounds, pipeline_state.xpos[self.payload_body_id][2] < 0.05)

    obs = self._get_obs(pipeline_state, prev_last_action, self.target_position)
    reward, _, _ = self.calc_reward(
        obs, pipeline_state.time, collision, out_of_bounds, action_scaled,
        angle_q1, angle_q2, prev_last_action, self.target_position, pipeline_state
    )
    done = jp.logical_or(jp.logical_or(out_of_bounds, collision),
                         pipeline_state.time > self.max_time)
    done = done * 1.0

    metrics = {'time': pipeline_state.time, 'reward': reward}
    return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward, done=done, metrics=metrics)

  def _get_obs(self, data: base.State, last_action: jp.ndarray, target_position: jp.ndarray) -> jp.ndarray:
    """Constructs the observation vector from simulation data."""
    # Payload state.
    payload_pos = data.xpos[self.payload_body_id]
    payload_linvel = data.cvel[self.payload_body_id][3:6]
    payload_error = target_position - payload_pos

    # Quad 1 state.
    quad1_pos = data.xpos[self.q1_body_id]
    quad1_quat = data.xquat[self.q1_body_id]
    quad1_linvel = data.cvel[self.q1_body_id][3:6]
    quad1_angvel = data.cvel[self.q1_body_id][:3]
    quad1_rel = quad1_pos - payload_pos
    quad1_rot = jp_R_from_quat(quad1_quat).ravel()
    quad1_linear_acc = data.cacc[self.q1_body_id][3:6]
    quad1_angular_acc = data.cacc[self.q1_body_id][:3]
    quad1_id = jp.array([1.0, 0.0])

    # Quad 2 state.
    quad2_pos = data.xpos[self.q2_body_id]
    quad2_quat = data.xquat[self.q2_body_id]
    quad2_linvel = data.cvel[self.q2_body_id][3:6]
    quad2_angvel = data.cvel[self.q2_body_id][:3]
    quad2_rel = quad2_pos - payload_pos
    quad2_rot = jp_R_from_quat(quad2_quat).ravel()
    quad2_linear_acc = data.cacc[self.q2_body_id][3:6]
    quad2_angular_acc = data.cacc[self.q2_body_id][:3]
    quad2_id = jp.array([0.0, 1.0])

    # # local-frame observations for each quad.
    # # For quad1:
    # R1 = jp_R_from_quat(quad1_quat)      # rotation matrix: local -> global
    # R1_T = jp.transpose(R1)              # global -> local
    # local_quad1_rel         = jp.matmul(R1_T, quad1_rel)
    # local_quad1_linvel      = jp.matmul(R1_T, quad1_linvel)
    # local_quad1_angvel      = jp.matmul(R1_T, quad1_angvel)
    # local_quad1_linear_acc  = jp.matmul(R1_T, quad1_linear_acc)
    # local_quad1_angular_acc = jp.matmul(R1_T, quad1_angular_acc)
    # q1_q2_rel = quad2_pos - quad1_pos
    # local_q1_q2_rel = jp.matmul(R1_T, q1_q2_rel)
    # local_q1_payload_error = jp.matmul(R1_T, payload_error)
    # local_q1_payload_linvel = jp.matmul(R1_T, payload_linvel)

    # # For quad2:
    # R2 = jp_R_from_quat(quad2_quat)
    # R2_T = jp.transpose(R2)
    # local_quad2_rel         = jp.matmul(R2_T, quad2_rel)
    # local_quad2_linvel      = jp.matmul(R2_T, quad2_linvel)
    # local_quad2_angvel      = jp.matmul(R2_T, quad2_angvel)
    # local_quad2_linear_acc  = jp.matmul(R2_T, quad2_linear_acc)
    # local_quad2_angular_acc = jp.matmul(R2_T, quad2_angular_acc)
    # local_q2_q1_rel = jp.matmul(R2_T, -q1_q2_rel)
    # local_q2_payload_error = jp.matmul(R2_T, payload_error)
    # local_q2_payload_linvel = jp.matmul(R2_T, payload_linvel)
    
    # # Helper function to convert Cartesian to spherical coordinates.
    # def cartesian_to_spherical(vec):
    #     r = jp.linalg.norm(vec) + 1e-6
    #     theta = jp.arccos(vec[2] / r)
    #     phi = jp.arctan2(vec[1], vec[0])
    #     return jp.array([r, theta, phi])
    
    # # Compute spherical coordinates for the relevant local vectors.
    # sph_local_quad1_rel = cartesian_to_spherical(local_quad1_rel)
    # sph_local_quad2_rel = cartesian_to_spherical(local_quad2_rel)
    # sph_local_q1_payload_error = cartesian_to_spherical(local_q1_payload_error)
    # sph_local_q2_payload_error = cartesian_to_spherical(local_q2_payload_error)
    # sph_local_q1_q2_rel = cartesian_to_spherical(local_q1_q2_rel)
    # sph_local_q2_q1_rel = cartesian_to_spherical(local_q2_q1_rel)

    obs = jp.concatenate([
      # ----                  # Shape  Index
        payload_error,        # (3,)  0-2
        payload_linvel,       # (3,)  3-5
        quad1_rel,            # (3,)  6-8
        quad1_rot,            # (9,)  9-17
        quad1_linvel,         # (3,)  18-20
        quad1_angvel,         # (3,)  21-23
        quad1_linear_acc,     # (3,)  24-26
        quad1_angular_acc,    # (3,)  27-29
        quad2_rel,            # (3,)  30-32
        quad2_rot,            # (9,)  33-41
        quad2_linvel,         # (3,)  42-44
        quad2_angvel,         # (3,)  45-47
        quad2_linear_acc,     # (3,)  48-50
        quad2_angular_acc,    # (3,)  51-53
        last_action,          # (8,)  54-61
        # local_quad1_rel,      # (3,)  62-64
        # local_quad1_linvel,   # (3,)  65-67
        # local_quad1_angvel,   # (3,)  68-70
        # local_quad1_linear_acc, # (3,) 71-73
        # local_quad1_angular_acc, # (3,) 74-76
        # local_quad2_rel,      # (3,)  77-79
        # local_quad2_linvel,   # (3,)  80-82
        # local_quad2_angvel,   # (3,)  83-85
        # local_quad2_linear_acc, # (3,) 86-88
        # local_quad2_angular_acc, # (3,) 89-91
        # local_q1_q2_rel,      # (3,)  92-94
        # local_q2_q1_rel,      # (3,)  95-97
        # local_q1_payload_error, # (3,) 98-100
        # local_q1_payload_linvel, # (3,) 101-103
        # local_q2_payload_error, # (3,) 104-106
        # local_q2_payload_linvel, # (3,) 107-109
        # sph_local_quad1_rel,        # (3,) 110-112
        # sph_local_quad2_rel,        # (3,) 113-115
        # sph_local_q1_payload_error, # (3,) 116-118
        # sph_local_q2_payload_error, # (3,) 119-121
        # sph_local_q1_q2_rel,        # (3,) 122-124
        # sph_local_q2_q1_rel,        # (3,) 125-127
    ])

    return obs

  def calc_reward(self, obs, sim_time, collision, out_of_bounds, action,
                  angle_q1, angle_q2, last_action, target_position, data) -> (jp.ndarray, None, dict):
    """
    Computes the reward by combining several factors such as payload tracking, quad safety,
    and energy penalties.
    """
    # Team observations: payload error and linear velocity.
    team_obs = obs[:6]
    payload_error = team_obs[:3]
    payload_linvel = team_obs[3:6]
    linvel_penalty = jp.linalg.norm(payload_linvel)
    dis = jp.linalg.norm(payload_error)
    z_error = jp.abs(payload_error[2])
    distance_reward = (1 - dis + jp.exp(-10 * dis)) + jp.exp(-10 * z_error) - z_error**2

    # Velocity alignment.
    norm_error = jp.maximum(jp.linalg.norm(payload_error), 1e-6)
    norm_linvel = jp.maximum(jp.linalg.norm(payload_linvel), 1e-6)
    velocity_towards_target = jp.dot(payload_error, payload_linvel) / (norm_error * norm_linvel)

    # Safety and smoothness penalties.
    quad1_obs = obs[6:30]
    quad2_obs = obs[30:54]
    quad_distance = jp.linalg.norm(quad1_obs[:3] - quad2_obs[:3])
    safe_distance_reward = jp.clip((quad_distance - 0.12) / (0.2 - 0.12), 0, 1)
    collision_penalty = 10.0 * collision
    out_of_bounds_penalty = 50.0 * out_of_bounds
    smooth_action_penalty = jp.mean(jp.abs(action - last_action) / self.max_thrust)
    action_energy_penalty = jp.mean(jp.abs(action)) / self.max_thrust

    # Reward for quad orientations (encouraging them to remain upright).
    up_reward = jp.exp(-jp.abs(angle_q1)) + jp.exp(-jp.abs(angle_q2))

    # Penalties for quad angular and linear velocities.
    ang_vel_q1 = quad1_obs[15:18]
    ang_vel_q2 = quad2_obs[15:18]
    ang_vel_penalty = 0.1 * (jp.linalg.norm(ang_vel_q1)**2 + jp.linalg.norm(ang_vel_q2)**2)
    linvel_q1 = quad1_obs[9:12]
    linvel_q2 = quad2_obs[9:12]
    linvel_quad_penalty = 0.1 * (jp.linalg.norm(linvel_q1)**2 + jp.linalg.norm(linvel_q2)**2)

    reward = 0
    reward += 10 * distance_reward 
    reward += safe_distance_reward
    reward += velocity_towards_target
    reward += up_reward
    #reward += 100 * quad_distance
    reward -= 10 * linvel_penalty
    reward -= collision_penalty
    reward -= out_of_bounds_penalty
    reward -= 2 * smooth_action_penalty
    reward -= action_energy_penalty
    reward -= ang_vel_penalty
    reward -= 5 * linvel_quad_penalty
    reward /= 25.0

    return reward, None, {}

# Register the environment under the name 'multiquad'
envs.register_environment('multiquad', MultiQuadEnv)