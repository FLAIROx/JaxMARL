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
      episode_length: int = 8192,                  # Maximum simulation time per episode.
      reward_coeffs: dict = None,
      obs_noise: float = 0.0,           # Parameter for observation noise
      act_noise: float = 0.0,         # Parameter for actuator noise
      max_thrust_range: float = 0.3,               # range for randomizing thrust
      **kwargs,
  ):
    print("Initializing MultiQuadEnv")
    # Load the MJX model from the XML file.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(current_dir, "mujoco", "three_quad_payload.xml")
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    # Convert the MuJoCo model to a Brax system.
    sys = mjcf.load_model(mj_model)
    kwargs['n_frames'] = kwargs.get('n_frames', sim_steps_per_action)
    kwargs['backend'] = 'mjx'
    super().__init__(sys, **kwargs)

    self.cable_length = 0.3

    # Save environment parameters.
    self.policy_freq = policy_freq
    self.sim_steps_per_action = sim_steps_per_action
    self.time_per_action = 1.0 / self.policy_freq
    self.max_time = episode_length * self.time_per_action
    self.obs_noise = obs_noise    
    self.act_noise = act_noise 
    if reward_coeffs is None:
      reward_coeffs = {
         "distance_reward_coef": 0.0,
         "z_distance_reward_coef": 0.0,
         "velocity_reward_coef": 0.0,
         "safe_distance_coef": 1.0,
         "up_reward_coef": 1.0,
         "linvel_reward_coef": 0.0,
         "ang_vel_reward_coef": 0.0,
         "linvel_quad_reward_coef": 1.0,
         "taut_reward_coef": 1.0,
         "collision_penalty_coef": -1.0,
         "out_of_bounds_penalty_coef": -1.0,
         "smooth_action_coef": -1.0,
         "action_energy_coef": 0.0,
      }

    self.reward_divisor = sum(reward_coeffs.values())
    self.reward_coeffs = reward_coeffs
    print("Reward coefficients:", self.reward_coeffs)

    self.warmup_time = 1.0

    # Set simulation timestep based on policy frequency and steps per action.
    dt = self.time_per_action / self.sim_steps_per_action
    sys.mj_model.opt.timestep = dt

    # Maximum thrust from the original environment.
    self.base_max_thrust = 0.14
    self.max_thrust_range = max_thrust_range
    # Define the target goal for the payload.
    self.goal_center = jp.array([0.0, 0.0, 1.5])
    self.target_position = self.goal_center

    # Cache body IDs (if still needed)
    self.payload_body_id = mujoco.mj_name2id(
        sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, "payload")
    self.q1_body_id = mujoco.mj_name2id(
        sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, "q0_container")
    self.q2_body_id = mujoco.mj_name2id(
        sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, "q1_container")
    self.q3_body_id = mujoco.mj_name2id(
        sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, "q2_container")

    # Cache joint IDs using the new API.
    self.payload_joint_id = mujoco.mj_name2id(
        sys.mj_model, mujoco.mjtObj.mjOBJ_JOINT.value, "payload_joint")
    self.q1_joint_id = mujoco.mj_name2id(
        sys.mj_model, mujoco.mjtObj.mjOBJ_JOINT.value, "q0_joint")
    self.q2_joint_id = mujoco.mj_name2id(
        sys.mj_model, mujoco.mjtObj.mjOBJ_JOINT.value, "q1_joint")
    self.q3_joint_id = mujoco.mj_name2id(
        sys.mj_model, mujoco.mjtObj.mjOBJ_JOINT.value, "q2_joint")

    # Cache the starting indices in qpos from the model.
    self.payload_qpos_start = sys.mj_model.jnt_qposadr[self.payload_joint_id]
    self.q1_qpos_start = sys.mj_model.jnt_qposadr[self.q1_joint_id]
    self.q2_qpos_start = sys.mj_model.jnt_qposadr[self.q2_joint_id]
    self.q3_qpos_start = sys.mj_model.jnt_qposadr[self.q3_joint_id]


    
    print("IDs:")
    print("Payload body ID:", self.payload_body_id)
    print("Quad 1 body ID:", self.q1_body_id)
    print("Quad 2 body ID:", self.q2_body_id)
    print("Quad 3 body ID:", self.q3_body_id)
    print("Payload joint ID:", self.payload_joint_id)
    print("Quad 1 joint ID:", self.q1_joint_id)
    print("Quad 2 joint ID:", self.q2_joint_id)
    print("Quad 3 joint ID:", self.q3_joint_id)
    print("Payload qpos start:", self.payload_qpos_start)
    print("Quad 1 qpos start:", self.q1_qpos_start)
    print("Quad 2 qpos start:", self.q2_qpos_start)
    print("Quad 3 qpos start:", self.q3_qpos_start)
    print("Noise_level:", self.obs_noise)
    print("MultiQuadEnv initialized successfully.")

        # Throw error if any ids are not found in a short way
    if self.payload_body_id == -1 or self.q1_body_id == -1 or self.q2_body_id == -1 or self.q3_body_id == -1:
      raise ValueError("One or more body IDs not found in the model.")
    if self.payload_joint_id == -1 or self.q1_joint_id == -1 or self.q2_joint_id == -1 or self.q3_joint_id == -1:
      raise ValueError("One or more joint IDs not found in the model.")

  @staticmethod
  def generate_configuration(key, target_position):
    subkeys = jax.random.split(key, 11)
    min_quad_z = 0.008 # quad on ground
    min_payload_z = 0.0055 # payload on ground

    payload_xy = jax.random.uniform(subkeys[0], (2,), minval=-1.5, maxval=1.5)
    payload_z = jax.random.uniform(subkeys[1], (), minval=-1.0, maxval=3.0)
    uniform_payload_pos = jp.array([payload_xy[0], payload_xy[1], payload_z])
    
  
    # mask: if True use uniform sample, if False use normal sample.
    mask = jax.random.uniform(subkeys[9], (), minval=0.0, maxval=1.0) < 0.9
    normal_payload_pos = target_position + jax.random.normal(subkeys[10], (3,)) * 0.1
    
    # Choose payload position based on mask.
    payload_pos = jp.where(mask, uniform_payload_pos, normal_payload_pos)

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
    payload_pos = payload_pos.at[2].set(jp.clip(payload_pos[2], min_payload_z, 3))
    
    return payload_pos, quad1_pos, quad2_pos

  @staticmethod
  def generate_valid_configuration(key, target_position, oversample=5):
    candidate_keys = jax.random.split(key, oversample)
    # Use in_axes=(0, None) to broadcast target_position.
    candidate_payload, candidate_quad1, candidate_quad2 = jax.vmap(
        MultiQuadEnv.generate_configuration, in_axes=(0, None)
    )(candidate_keys, target_position)
    dist_quads = jp.linalg.norm(candidate_quad1 - candidate_quad2, axis=1)
    dist_q1_payload = jp.linalg.norm(candidate_quad1 - candidate_payload, axis=1)
    dist_q2_payload = jp.linalg.norm(candidate_quad2 - candidate_payload, axis=1)
    
    valid_mask = (dist_quads >= 0.16) & (dist_q1_payload >= 0.07) & (dist_q2_payload >= 0.07)
    valid_index = jp.argmax(valid_mask)  # returns 0 if none are valid
    
    return candidate_payload[valid_index], candidate_quad1[valid_index], candidate_quad2[valid_index]

  def reset(self, rng: jax.Array) -> State:
    """Resets the environment to an initial state."""
    # randomize max_thrust between in range
    rng, mt_rng = jax.random.split(rng)
    factor = jax.random.uniform(mt_rng, (), minval=1.0 - self.max_thrust_range, maxval=1.0)
    max_thrust = self.base_max_thrust * factor
    
    rng, rng1, rng2, rng_config = jax.random.split(rng, 4)

    base_qpos = self.sys.qpos0  # Start with the reference configuration.
    qvel = 0.1 * jax.random.normal(rng2, (self.sys.nv,))
    qvel = jp.clip(qvel, a_min=-5.0, a_max=5.0)

    # Get new positions for payload and both quadrotors.
    payload_pos, quad1_pos, quad2_pos = MultiQuadEnv.generate_valid_configuration(rng_config, self.target_position)

    # Generate new orientations (as quaternions) for the quadrotors.
    rng, rng_euler = jax.random.split(rng, 2)
    keys = jax.random.split(rng_euler, 6)
    std_dev = 10 * jp.pi / 180   # 5° in radians.
    clip_val = 60 * jp.pi / 180  # 60° in radians.

    # Quadrotor 1: sample roll and pitch (clipped) and yaw uniformly.
    roll_q1 = jp.clip(jax.random.normal(keys[0]) * std_dev, -clip_val, clip_val)
    pitch_q1 = jp.clip(jax.random.normal(keys[1]) * std_dev, -clip_val, clip_val)
    yaw_q1 = jax.random.uniform(keys[2], minval=-jp.pi, maxval=jp.pi)
    # Quadrotor 2: similarly.
    roll_q2 = jp.clip(jax.random.normal(keys[3]) * std_dev, -clip_val, clip_val)
    pitch_q2 = jp.clip(jax.random.normal(keys[4]) * std_dev, -clip_val, clip_val)
    yaw_q2 = jax.random.uniform(keys[5], minval=-jp.pi, maxval=jp.pi)

    # Set roll and pitch to 0 if the quad's z value is below 0.01.
    roll_q1 = jp.where(quad1_pos[2] < 0.02, 0.0, roll_q1)
    pitch_q1 = jp.where(quad1_pos[2] < 0.02, 0.0, pitch_q1)
    roll_q2 = jp.where(quad2_pos[2] < 0.02, 0.0, roll_q2)
    pitch_q2 = jp.where(quad2_pos[2] < 0.02, 0.0, pitch_q2)

    def euler_to_quat(roll, pitch, yaw):
      cr = jp.cos(roll * 0.5)
      sr = jp.sin(roll * 0.5)
      cp = jp.cos(pitch * 0.5)
      sp = jp.sin(pitch * 0.5)
      cy = jp.cos(yaw * 0.5)
      sy = jp.sin(yaw * 0.5)
      # MuJoCo now expects quaternions in (w, x, y, z) order.
      return jp.array([
          cr * cp * cy + sr * sp * sy,  # w
          sr * cp * cy - cr * sp * sy,  # x
          cr * sp * cy + sr * cp * sy,  # y
          cr * cp * sy - sr * sp * cy,  # z
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
    
    
    pipeline_state = self.pipeline_init(new_qpos, qvel)
    last_action = jp.zeros(self.sys.nu)
    rng, noise_key = jax.random.split(rng)       # new: split for observation noise
    obs = self._get_obs(pipeline_state, last_action, self.target_position, noise_key)
    reward = jp.array(0.0)
    done = jp.array(0.0)
    metrics = {'time': pipeline_state.time,
               'reward': reward,
               'max_thrust': max_thrust}
    return State(pipeline_state, obs, reward, done, metrics)

  
  def step(self, state: State, action: jax.Array) -> State:
    """Advances the environment by one control step.
    
    Args:
        state: The current state.
        action: The action to apply.
    
    Returns:
        The next state.
    """
    # Extract previous action from the observation.
    prev_last_action = state.obs[-self.sys.nu:]
    # Scale actions from [-1, 1] to thrust commands in [0, max_thrust].
    max_thrust = state.metrics['max_thrust']
    thrust_cmds = 0.5 * (action + 1.0)
    action_scaled = thrust_cmds * max_thrust

    data0 = state.pipeline_state
    pipeline_state = self.pipeline_step(data0, action_scaled)

    # Generate a dynamic noise_key using pipeline_state fields.
    noise_key = jax.random.PRNGKey(0)
    noise_key = jax.random.fold_in(noise_key, jp.int32(pipeline_state.time * 1e6))
    noise_key = jax.random.fold_in(noise_key, jp.int32(jp.sum(pipeline_state.xpos) * 1e3))
    noise_key = jax.random.fold_in(noise_key, jp.int32(jp.sum(pipeline_state.cvel) * 1e3))

    # Add actuator noise.
    if self.act_noise:
        noise = jax.random.normal(noise_key, shape=action_scaled.shape)
        action_scaled = action_scaled + self.act_noise * max_thrust * noise

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
    
    quad_collision = quad_distance < 0.15 # quad is square with 5cm so radius is 0.0707m

    ground_collision_quad = jp.logical_and(pipeline_state.xpos[self.q2_body_id][2] < 0.03, pipeline_state.xpos[self.q1_body_id][2] < 0.03)
    ground_collision_payload = pipeline_state.xpos[self.payload_body_id][2] < 0.03
    
    ground_collision = jp.logical_and(ground_collision_quad, ground_collision_payload)

    collision = jp.logical_or(quad_collision, ground_collision)

    out_of_bounds = jp.logical_or(jp.absolute(angle_q1) > jp.radians(90),
                                  jp.absolute(angle_q2) > jp.radians(90))
    out_of_bounds = jp.logical_or(out_of_bounds, pipeline_state.xpos[self.q1_body_id][2] < pipeline_state.xpos[self.payload_body_id][2]-0.05)
    out_of_bounds = jp.logical_or(out_of_bounds, pipeline_state.xpos[self.q2_body_id][2] < pipeline_state.xpos[self.payload_body_id][2]-0.05)



  
    #out of bounds for pos error shrinking with time
    payload_pos = pipeline_state.xpos[self.payload_body_id]
    payload_error = self.target_position - payload_pos
    payload_error_norm = jp.linalg.norm(payload_error)
    max_time_to_target = self.max_time * 0.75
    time_progress = jp.clip(pipeline_state.time / max_time_to_target, 0.0, 1.0)
    max_payload_error = 4 * (1 - time_progress) + 0.05 # allow for 5cm error at the target
    out_of_bounds = jp.logical_or(out_of_bounds, payload_error_norm > max_payload_error)



    obs = self._get_obs(pipeline_state, prev_last_action, self.target_position, noise_key)
    reward, _, _ = self.calc_reward(
        obs, pipeline_state.time, collision, out_of_bounds, action_scaled,
        angle_q1, angle_q2, prev_last_action, self.target_position,
        pipeline_state, max_thrust
    )

    # dont terminate ground collision on ground start
    ground_collision = jp.logical_and(
      ground_collision,
      jp.logical_or(
        pipeline_state.time > 3,
        pipeline_state.cvel[self.payload_body_id][2] < -3.0,
      )
    )

    collision = jp.logical_or(quad_collision, ground_collision)
    
    done = jp.logical_or(jp.logical_or(out_of_bounds, collision),
                         pipeline_state.time > self.max_time*1.2) # this should never happen, because episode ends first
   
    
    done = done * 1.0

    metrics = {
      'time': pipeline_state.time,
      'reward': reward,
      'max_thrust': state.metrics['max_thrust']
    }
    return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward, done=done, metrics=metrics)

  def _get_obs(self, data: base.State, last_action: jp.ndarray, target_position: jp.ndarray, noise_key) -> jp.ndarray:
    """Constructs the observation vector from simulation data."""
    # Payload state.
    payload_pos = data.xpos[self.payload_body_id]    
    payload_linvel = data.cvel[self.payload_body_id][3:6]
    payload_error = target_position - payload_pos
    distance = jp.linalg.norm(payload_error)
    payload_error = payload_error / jp.maximum(distance, 1.0)  # Normalize if distance > 1

    # Convert payload_error from Cartesian to spherical global coordinates.
    def cartesian_to_spherical(vec):
        r = jp.linalg.norm(vec)
        theta = jp.arccos(vec[2] / (r + 1e-6))
        phi = jp.arctan2(vec[1], vec[0])
        return jp.array([r, theta, phi])
    #payload_error_sph = cartesian_to_spherical(payload_error)

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

    # Lookup for noise scale factors (each multiplied with self.obs_noise):
    noise_lookup = jp.concatenate([
        jp.ones(3) * 0.005,  # indices 0-2: payload_error noise scale
        jp.ones(3) * 0.05,  # indices 3-5: payload_linvel noise scale
        jp.ones(3) * 0.02,  # indices 6-8: quad1_rel noise scale
        jp.ones(9) * 0.005, # indices 9-17: quad1_rot noise scale
        jp.ones(3) * 0.05,  # indices 18-20: quad1_linvel noise scale
        jp.ones(3) * 0.08,  # indices 21-23: quad1_angvel noise scale
        jp.ones(3) * 0.05,  # indices 24-26: quad1_linear_acc noise scale
        jp.ones(3) * 0.05,  # indices 27-29: quad1_angular_acc noise scale
        jp.ones(3) * 0.02,  # indices 30-32: quad2_rel noise scale
        jp.ones(9) * 0.005, # indices 33-41: quad2_rot noise scale
        jp.ones(3) * 0.05,  # indices 42-44: quad2_linvel noise scale
        jp.ones(3) * 0.08,  # indices 45-47: quad2_angvel noise scale
        jp.ones(3) * 0.05,  # indices 48-50: quad2_linear_acc noise scale
        jp.ones(3) * 0.05,  # indices 51-53: quad2_angular_acc noise scale
        jp.ones(8) * 0.0,   # indices 54-61: last_action noise scale
    ])

    if self.obs_noise != 0.0:
        noise = self.obs_noise * noise_lookup * jax.random.normal(noise_key, shape=obs.shape)
        obs = obs + noise
    return obs

  def calc_reward(self, obs, sim_time, collision, out_of_bounds, action,
                  angle_q1, angle_q2, last_action, target_position, data,
                  max_thrust) -> (jp.ndarray, None, dict):
    """
    Computes the reward by combining several factors such as payload tracking, quad safety,
    and energy penalties.
    """
    # lambda for exponential reward
    er = lambda x, s=2: jp.exp(-s * jp.abs(x))

    # Team observations: payload error and linear velocity.
    team_obs = obs[:6]
    payload_error = team_obs[:3]
    payload_linvel = team_obs[3:6]
    linvel_reward = er(jp.linalg.norm(payload_linvel))


    dis = jp.linalg.norm(payload_error)
    z_error = jp.abs(payload_error[2])
    distance_reward =  er(dis)#, 2 + sim_time)
    z_distance_reward =  er(z_error)#, 2 + sim_time)

    


    # Safety and smoothness penalties.
    quad1_obs = obs[6:29]
    quad2_obs = obs[30:54]
    quad_distance = jp.linalg.norm(quad1_obs[:3] - quad2_obs[:3])
    safe_distance_reward = jp.clip((quad_distance - 0.15) / (0.17 - 0.15), 0, 1)
    collision_penalty = 1.0 * collision
    # out_of_bounds_penalty = 50.0 * out_of_bounds

    # Reward for quad orientations (encouraging them to remain upright).
    up_reward = 0.5 * er(angle_q1) + 0.5 * er(angle_q2)

    # taut string reward
    quad1_dist = jp.linalg.norm(quad1_obs[:3]) # payload to quad1
    quad2_dist = jp.linalg.norm(quad2_obs[:3]) # payload to quad2
    taut_reward = quad1_dist + quad2_dist # Maximize the string length
    taut_reward += quad1_obs[2] + quad2_obs[2] # Maximize the height of the quads
    taut_reward /= self.cable_length

    # Reward for quad velocities.
    # The reward is higher for lower angular velocities.
    # The reward is higher for lower linear velocities. 
    ang_vel_q1 = quad1_obs[15:18]
    ang_vel_q2 = quad2_obs[15:18]
    ang_vel_reward = (0.5 + 3 * er(dis, 20)) * (er(jp.linalg.norm(ang_vel_q1)) + er(jp.linalg.norm(ang_vel_q2)))
    linvel_q1 = quad1_obs[9:12]
    linvel_q2 = quad2_obs[9:12]
    linvel_quad_reward =  (0.5 + 6 * er(dis, 20)) * (er(jp.linalg.norm(linvel_q1)) + er(jp.linalg.norm(linvel_q2)) )

    # Velocity alignment.
    target_dir  = payload_error / (dis + 1e-6)
    vel = jp.linalg.norm(payload_linvel)
    # Avoid division by zero. 
    vel_dir = jp.where(jp.abs(vel) > 1e-6, payload_linvel / vel, jp.zeros_like(payload_linvel))
  

    aligned_vel = er(1 - jp.dot(vel_dir, target_dir), dis) # dotprod = 1  => vel is perfectly aligned
    velocity_towards_target = aligned_vel
  

    vel_cap = 3.45 - 0.115 * vel**4
    zero_at_target = 14.7 * dis * jp.exp(-10.5 * dis * jp.abs(vel))
    no_zero_while_error = jp.exp(- (0.5 / (26.0 * dis + 0.3)) * jp.abs(vel))
    target_reward = no_zero_while_error * (vel_cap - zero_at_target)
    target_reward = jp.exp(0.4 * target_reward)
    target_reward *= aligned_vel
    target_reward *= jp.exp(-1.4 * jp.abs(dis))


    smooth_action_penalty = jp.mean(jp.abs(action - last_action) / max_thrust)
    action_energy_penalty = jp.mean(jp.abs(action)) / max_thrust



    tracking_reward = self.reward_coeffs["distance_reward_coef"] * distance_reward
    #tracking_reward += self.reward_coeffs["z_distance_reward_coef"] * z_distance_reward
    tracking_reward += self.reward_coeffs["velocity_reward_coef"] * velocity_towards_target
    # tracking_reward += self.reward_coeffs.get("target_reward_coef", 1.0) * target_reward
    #tracking_reward = target_reward

    stability_reward = self.reward_coeffs["up_reward_coef"] * up_reward
    stability_reward += self.reward_coeffs["ang_vel_reward_coef"] * ang_vel_reward
    stability_reward += self.reward_coeffs["linvel_reward_coef"] * linvel_reward
    stability_reward += self.reward_coeffs["linvel_quad_reward_coef"] * linvel_quad_reward
    stability_reward += self.reward_coeffs["taut_reward_coef"] * taut_reward
    

    #penalties
    safety_reward = self.reward_coeffs["collision_penalty_coef"] * collision_penalty
    safety_reward += self.reward_coeffs["out_of_bounds_penalty_coef"] * out_of_bounds
    safety_reward += self.reward_coeffs["smooth_action_coef"] * smooth_action_penalty
    safety_reward += self.reward_coeffs["action_energy_coef"] * action_energy_penalty
    safety_reward += self.reward_coeffs["safe_distance_coef"] * safe_distance_reward

  
    # Combine all rewards and penalties.
   
    reward = tracking_reward * (stability_reward + safety_reward)
    
    # Normalize the reward by the divisor.
    reward /= self.reward_divisor

    return reward, None, {}

# Register the environment under the name 'threequad'
envs.register_environment('threequad', MultiQuadEnv)