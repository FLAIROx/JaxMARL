# quad_env.py

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

# Helper functions in JAX
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


class QuadEnv(PipelineEnv):
  """
  A Brax/MJX environment for a multi-rotor quadcopter team carrying a payload.

  The environment is initialized from a MuJoCo XML model and then converted into a Brax
  system using the MJX backend. The control actions (in [-1, 1]) are scaled into thrust commands.
  """
  # total obs dims: 3(pos_error)+9(rot)+3(linvel)+3(angvel)+3(lin_acc)+4(last_action)
  OBS_SIZE = 3 + 9 + 3 + 3 #+ 3 # +4

  def __init__(
      self,
      policy_freq: float = 250,              # Policy frequency in Hz.
      sim_steps_per_action: int = 1,           # Physics steps between control actions.
      episode_length: int = 3072,                  # Maximum simulation time per episode.
      reward_coeffs: dict = None,
      obs_noise: float = 0.0,           # Parameter for observation noise
      act_noise: float = 0.0,         # Parameter for actuator noise
      max_thrust_range: float = 0.2,               # range for randomizing thrust
      debug: bool = False,
      tau_up: float = 0.0,      
      tau_down: float = 0.0,    
      **kwargs,
  ):
    print("Initializing QuadEnv")
    # Load the MJX model from the XML file.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(current_dir, "mujoco", "quad.xml")
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
    self.debug = debug
    self.tau_up = tau_up           
    self.tau_down = tau_down       
   
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
    self.base_max_thrust = 0.13 # 0.118 in firmware
    self.max_thrust_range = max_thrust_range
    # Define the target goal for the payload.
    self.goal_center = jp.array([0.0, 0.0, 1.5])
    self.target_position = self.goal_center

    # Cache body IDs (if still needed)
    # self.payload_body_id = mujoco.mj_name2id(
    #     sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, "payload")
    self.q1_body_id = mujoco.mj_name2id(
        sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, "q0_container")

    # Cache joint IDs using the new API.
    # self.payload_joint_id = mujoco.mj_name2id(
    #     sys.mj_model, mujoco.mjtObj.mjOBJ_JOINT.value, "payload_joint")
    self.q1_joint_id = mujoco.mj_name2id(
        sys.mj_model, mujoco.mjtObj.mjOBJ_JOINT.value, "q0_joint")
   

    # Cache the starting indices in qpos from the model.
    # self.payload_qpos_start = sys.mj_model.jnt_qposadr[self.payload_joint_id]
    self.q1_qpos_start = sys.mj_model.jnt_qposadr[self.q1_joint_id]



    
    print("IDs:")
    # print("Payload body ID:", self.payload_body_id)
    print("Quad 1 body ID:", self.q1_body_id)
    # print("Quad 2 body ID:", self.q2_body_id)
    # print("Payload joint ID:", self.payload_joint_id)
    print("Quad 1 joint ID:", self.q1_joint_id)
    # print("Quad 2 joint ID:", self.q2_joint_id)
    # print("Payload qpos start:", self.payload_qpos_start)
    print("Quad 1 qpos start:", self.q1_qpos_start)
    # print("Quad 2 qpos start:", self.q2_qpos_start)
    print("Noise_level:", self.obs_noise)
    print("QuadEnv initialized successfully.")

        # Throw error if any ids are not found in a short way
    # if self.payload_body_id == -1 or self.q1_body_id == -1 or self.q2_body_id == -1:
    #   raise ValueError("One or more body IDs not found in the model.")
    # if self.payload_joint_id == -1 or self.q1_joint_id == -1 or self.q2_joint_id == -1:
    #   raise ValueError("One or more joint IDs not found in the model.")

    if self.q1_body_id == -1:
      raise ValueError("Quad 1 body ID not found in the model.")
    if self.q1_joint_id == -1:
      raise ValueError("Quad 1 joint ID not found in the model.")

  @staticmethod
  def generate_configuration(key, target_position):
    subkeys = jax.random.split(key, 11)
    min_quad_z = 0.008 # quad on ground
    min_payload_z = 0.0055 # payload on ground

    payload_xy = jax.random.uniform(subkeys[0], (2,), minval=-1.5, maxval=1.5)
    payload_z = jax.random.uniform(subkeys[1], (), minval=-1.0, maxval=3.0)
    uniform_payload_pos = jp.array([payload_xy[0], payload_xy[1], payload_z])
    
  
    # mask: if True use uniform sample, if False use normal sample.
    mask = jax.random.uniform(subkeys[9], (), minval=0.0, maxval=1.0) < 0.5
    normal_payload_pos = target_position + jax.random.normal(subkeys[10], (3,)) * 0.2
    
    # Choose payload position based on mask.
    payload_pos = jp.where(mask, uniform_payload_pos, normal_payload_pos)

    # Parameters for Quad positions.
    mean_r   = 0
    std_r    = 0
    clip_min = 0
    clip_max = 0.3
    mean_theta  = jp.pi / 7
    mean_theta2 = -jp.pi / 7
    std_theta   = jp.pi / 8
    std_phi     = jp.pi / 3

    # Quad 1.
    r1     = jp.clip(mean_r + std_r * jax.random.normal(subkeys[2], ()), clip_min, clip_max)
    theta1 = mean_theta + std_theta * jax.random.normal(subkeys[4], ())

    # Common phi offset and individual noise.
    phi_offset = jax.random.uniform(subkeys[6], (), minval=-jp.pi, maxval=jp.pi)
    phi1 = std_phi * jax.random.normal(subkeys[7], ()) + phi_offset

    # Convert spherical to Cartesian for Quad 1.
    quad1_x = r1 * jp.sin(theta1) * jp.cos(phi1) + payload_pos[0]
    quad1_y = r1 * jp.sin(theta1) * jp.sin(phi1) + payload_pos[1]
    quad1_z = jp.clip(r1 * jp.cos(theta1) + payload_pos[2], min_quad_z, 3)
    quad1_pos = jp.array([quad1_x, quad1_y, quad1_z])
    

    # Ensure payload is above ground.
    payload_pos = payload_pos.at[2].set(jp.clip(payload_pos[2], min_payload_z, 3))
    
    return payload_pos, quad1_pos

  @staticmethod
  def generate_valid_configuration(key, target_position, oversample=1):
    candidate_keys = jax.random.split(key, oversample)
    # Use in_axes=(0, None) to broadcast target_position.
    candidate_payload, candidate_quad1 = jax.vmap(
        QuadEnv.generate_configuration, in_axes=(0, None)
    )(candidate_keys, target_position)
    # dist_quads = jp.linalg.norm(candidate_quad1 - candidate_quad2, axis=1)
    # dist_q1_payload = jp.linalg.norm(candidate_quad1 - candidate_payload, axis=1)
    # dist_q2_payload = jp.linalg.norm(candidate_quad2 - candidate_payload, axis=1)
    
    # valid_mask = (dist_quads >= 0.16) & (dist_q1_payload >= 0.07)
    #valid_index = jp.argmax(valid_mask)  # returns 0 if none are valid
    
    return candidate_payload[0], candidate_quad1[0] #[valid_index], candidate_quad2[valid_index]

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
    payload_pos, quad1_pos = QuadEnv.generate_valid_configuration(rng_config, self.target_position)

    # 10% chance to reset quad to exactly the target
    rng, quad_reset_rng = jax.random.split(rng)
    quad_reset = jax.random.uniform(quad_reset_rng, (), minval=0.0, maxval=1.0) < 0.25
    quad1_pos = jp.where(quad_reset, self.target_position + jax.random.normal(rng, (3,)) * 0.005, quad1_pos)
    qvel = jp.where(quad_reset, 0.0, qvel)


    # Generate new orientations (as quaternions) for the quadrotors.
    rng, rng_euler = jax.random.split(rng, 2)
    keys = jax.random.split(rng_euler, 6)
    std_dev = 10 * jp.pi / 180   # 5° in radians.
    clip_val = 60 * jp.pi / 180  # 60° in radians.

    # Quadrotor 1: sample roll and pitch (clipped) and yaw uniformly.
    roll_q1 = jp.clip(jax.random.normal(keys[0]) * std_dev, -clip_val, clip_val)
    pitch_q1 = jp.clip(jax.random.normal(keys[1]) * std_dev, -clip_val, clip_val)
    yaw_q1 = jax.random.uniform(keys[2], minval=-jp.pi, maxval=jp.pi)


    # Set roll and pitch to 0 if the quad's z value is below 0.01.
    roll_q1 = jp.where(quad1_pos[2] < 0.02, 0.0, roll_q1)
    pitch_q1 = jp.where(quad1_pos[2] < 0.02, 0.0, pitch_q1)
   

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


    # Build the full qpos vector.
    new_qpos = base_qpos
    # Update payload position.
    #new_qpos = new_qpos.at[self.payload_qpos_start:self.payload_qpos_start+3].set(payload_pos)
    # Update quadrotor positions.
    new_qpos = new_qpos.at[self.q1_qpos_start:self.q1_qpos_start+3].set(quad1_pos)
    # Update quadrotor orientations (starting 3 elements later).
    new_qpos = new_qpos.at[self.q1_qpos_start+3:self.q1_qpos_start+7].set(quat_q1)
    
    
    pipeline_state = self.pipeline_init(new_qpos, qvel)
    last_action = jp.zeros(self.sys.nu)
    last_action = jax.random.normal(rng1, shape=last_action.shape) * 0.4
    last_action = jp.clip(last_action, -1.0, 1.0)

    rng, noise_key = jax.random.split(rng)       # new: split for observation noise
    obs = self._get_obs(
      pipeline_state,
      last_action,
      self.target_position,
      noise_key,
      prev_linvel=jp.zeros(3)
    )
    reward = jp.array(0.0)
    done = jp.array(0.0)

    # compute initial thrust and store it
    action_scaled = 0.5 * (last_action + 1.0)
    init_thrust = self.motor_model(action_scaled, max_thrust, self.act_noise, noise_key, last_thrust=None)

    metrics = {
      'time': pipeline_state.time,
      'reward': reward,
      'max_thrust': max_thrust,
      'last_action': last_action,
      'last_thrust': init_thrust,
    }
    return State(pipeline_state, obs, reward, done, metrics)
  
  def motor_model(self, action_normalized, max_thrust, act_noise, noise_key, last_thrust=None):
    """Motor model for thrust calculation."""
    # compute target thrust and corresponding PWM
    target_thrusts = jp.clip(action_normalized * max_thrust, 0.0, max_thrust)
    target_pwm = jp.sqrt(target_thrusts)


    if last_thrust is None or (self.tau_up <= 0.0 and self.tau_down <= 0.0):
      new_thrusts = target_thrusts
    else:
      # first-order dynamics on PWM
      dt = self.time_per_action
      last_pwm = jp.sqrt(jp.clip(last_thrust, 0.0, max_thrust))
      tau = jp.where(target_pwm >= last_pwm, self.tau_up, self.tau_down)
      pwm_filtered = last_pwm + (dt / tau) * (target_pwm - last_pwm)
      new_thrusts = jp.clip(pwm_filtered**2, 0.0, max_thrust)

    # apply actuator noise on thrust
    if act_noise > 0.0:
      noise = act_noise * max_thrust * jax.random.normal(noise_key, shape=new_thrusts.shape)
      new_thrusts = jp.clip(new_thrusts + noise, 0.0, max_thrust)
    return new_thrusts
  
  def step(self, state: State, action: jax.Array) -> State:
    """Advances the environment by one control step.
    
    Args:
        state: The current state.
        action: The action to apply.
    
    Returns:
        The next state.
    """
    # Extract previous action from the observation.
    last_action =  state.metrics['last_action']
    
    if self.debug:
      jax.debug.print("last_action: {last_action}", last_action=last_action)
      jax.debug.print("action: {action}", action=action)

    clipped_action = jp.clip(action, -1.0, 1.0)

    # Scale actions from [-1, 1] to thrust commands in [0, max_thrust].
    max_thrust = state.metrics['max_thrust']

     # Generate a dynamic noise_key using pipeline_state fields.
    noise_key = jax.random.PRNGKey(0)
    noise_key = jax.random.fold_in(noise_key, jp.int32(jp.sum(state.pipeline_state.time)))
    noise_key = jax.random.fold_in(noise_key, jp.int32(jp.sum(state.obs[:6])))

    # get previous thrust and pass to motor model
    prev_thrust = state.metrics.get('last_thrust', jp.zeros_like(action))
    action_scaled = 0.5 * (clipped_action + 1.0)
    motor_thusts_N = self.motor_model(
      action_normalized=action_scaled,
      max_thrust=max_thrust,
      act_noise=self.act_noise,
      noise_key=noise_key,
      last_thrust=prev_thrust
    )

    data0 = state.pipeline_state
    pipeline_state = self.pipeline_step(data0, motor_thusts_N)

   
    

    # Compute orientation and collision/out-of-bound checks.
    q1_orientation = pipeline_state.xquat[self.q1_body_id]
    up = jp.array([0.0, 0.0, 1.0])
    q1_local_up = jp_R_from_quat(q1_orientation)[:, 2]
    angle_q1 = jp_angle_between(q1_local_up, up)

    quad1_pos = pipeline_state.xpos[self.q1_body_id]
    
    

    ground_collision_quad = pipeline_state.xpos[self.q1_body_id][2] < 0.03
    # ground_collision_payload = pipeline_state.xpos[self.payload_body_id][2] < 0.03
    
    # ground_collision = jp.logical_and(ground_collision_quad, ground_collision_payload)
    ground_collision = ground_collision_quad

    collision = ground_collision_quad

    out_of_bounds = jp.absolute(angle_q1) > jp.radians(90)
                             
    # out_of_bounds = jp.logical_or(out_of_bounds, pipeline_state.xpos[self.q1_body_id][2] < pipeline_state.xpos[self.payload_body_id][2]-0.05)



  
    #out of bounds for pos error shrinking with time
    quad_error = self.target_position - quad1_pos
    quad_error_norm = jp.linalg.norm(quad_error)
    max_time_to_target = self.max_time * 0.5
    time_progress = jp.clip(pipeline_state.time / max_time_to_target, 0.0, 1.0)
    max_quad_error = 4 * (1 - time_progress) + 0.01 # allow for 1cm error at the target
    out_of_bounds = jp.logical_or(out_of_bounds, quad_error_norm > max_quad_error)



    # payload_pos = pipeline_state.xpos[self.payload_body_id]
    # payload_error = self.target_position - payload_pos
    # payload_error_norm = jp.linalg.norm(payload_error)
    # max_time_to_target = self.max_time * 0.75
    # time_progress = jp.clip(pipeline_state.time / max_time_to_target, 0.0, 1.0)
    # max_payload_error = 4 * (1 - time_progress) + 0.05 # allow for 5cm error at the target
    # out_of_bounds = jp.logical_or(out_of_bounds, payload_error_norm > max_payload_error)



    prev_linvel = data0.cvel[self.q1_body_id][3:6]
    obs = self._get_obs(
      pipeline_state,
      action,
      self.target_position,
      noise_key,
      prev_linvel
    )
    reward, _, _ = self.calc_reward(
        obs, pipeline_state.time, collision, out_of_bounds, action,
        angle_q1, last_action, self.target_position,
        pipeline_state, max_thrust
    )

    # dont terminate ground collision on ground start
    ground_collision = jp.logical_and(
      ground_collision,
      pipeline_state.time > 0.3
      # jp.logical_or(
      #   pipeline_state.time > 3,
      #   pipeline_state.cvel[self.payload_body_id][2] < -3.0,
      # )
    )

    collision = ground_collision
    
    done = jp.logical_or(jp.logical_or(out_of_bounds, collision),
                         pipeline_state.time > self.max_time*1.2) # this should never happen, because episode ends first
   
    
    done = done * 1.0

    metrics = {
      'time': pipeline_state.time,
      'reward': reward,
      'max_thrust': state.metrics['max_thrust'],
      'last_action': clipped_action,
      'last_thrust': motor_thusts_N,
    }
    if self.debug:
      jax.debug.print("---------")
    return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward, done=done, metrics=metrics)

  def _get_obs(self,
               data: base.State,
               last_action: jp.ndarray,
               target_position: jp.ndarray,
               noise_key,
               prev_linvel: jp.ndarray = None
              ) -> jp.ndarray:
    """Constructs the observation vector from simulation data."""
    # Payload state.
    #payload_pos = data.xpos[self.payload_body_id]    
    #payload_linvel = data.cvel[self.payload_body_id][3:6]
    #payload_error = target_position - payload_pos
    # distance = jp.linalg.norm(payload_error)
    # payload_error = payload_error / jp.maximum(distance, 1.0)  # Normalize if distance > 1

    # Convert payload_error from Cartesian to spherical global coordinates.
    def cartesian_to_spherical(vec):
        r = jp.linalg.norm(vec)
        theta = jp.arccos(vec[2] / (r + 1e-6))
        phi = jp.arctan2(vec[1], vec[0])
        return jp.array([r, theta, phi])
    #payload_error_sph = cartesian_to_spherical(payload_error)

    # Quad 1 state.
    quad1_pos = data.xpos[self.q1_body_id]
    pos_error = target_position - quad1_pos
    distance = jp.linalg.norm(pos_error)
    pos_error = pos_error / jp.maximum(distance, 1.0)  # Normalize if distance > 1
    quad1_quat = data.xquat[self.q1_body_id]
    quad1_linvel = data.cvel[self.q1_body_id][3:6]
    quad1_angvel = data.cvel[self.q1_body_id][:3]
    # quad1_rel = quad1_pos - payload_pos
    quad1_rot = jp_R_from_quat(quad1_quat).ravel()
    if prev_linvel is None:
      quad1_linear_acc = jp.zeros(3)
    else:
      curr_linvel = data.cvel[self.q1_body_id][3:6]
      quad1_linear_acc = (curr_linvel - prev_linvel) / self.time_per_action
    quad1_id = jp.array([1.0, 0.0])

    

    obs = jp.concatenate([
      # ----                  # Shape  Slice
        pos_error,            # (3,)   0:3
        quad1_rot,            # (9,)   3:12
        quad1_linvel,         # (3,)  12:15
        quad1_angvel,         # (3,)  15:18
        #quad1_linear_acc,     # (3,)  18:21
        #last_action,          # (4,)  21:25
    ])

    if self.debug:
      jax.debug.print("lpos_error: {pos_error}", pos_error=pos_error)
      jax.debug.print("quad1_rot: {quad1_rot}", quad1_rot=quad1_rot)
      jax.debug.print("quad1_linvel: {quad1_linvel}", quad1_linvel=quad1_linvel)
      jax.debug.print("quad1_angvel: {quad1_angvel}", quad1_angvel=quad1_angvel)
      jax.debug.print("quad1_linear_acc: {quad1_linear_acc}", quad1_linear_acc=quad1_linear_acc)
     # jax.debug.print("last_action: {last_action}", last_action=last_action)


    # runtime check
    assert obs.shape[0] == self.OBS_SIZE, f"obs length {obs.shape[0]} != expected {self.OBS_SIZE}"

    # Lookup for noise scale factors (each multiplied with self.obs_noise):
    noise_lookup = jp.concatenate([
        jp.ones(3) * 0.002,  # quad position
        jp.ones(9) * 0.01,   # quad rotation
        jp.ones(3) * 0.05,   # quad linear velocity
        jp.ones(3) * 0.05,   # quad angular velocity
       # jp.ones(3) * 0.05,   # quad linear acceleration
       # jp.ones(4) * 0.0,    # last action
    ])

    if self.obs_noise != 0.0:
        noise = self.obs_noise * noise_lookup * jax.random.normal(noise_key, shape=obs.shape)
        obs = obs + noise
    return obs

  def calc_reward(self, obs, sim_time, collision, out_of_bounds, action,
                  angle_q1, last_action, target_position, data,
                  max_thrust) -> (jp.ndarray, None, dict):
    """
    Computes the reward by combining several factors such as payload tracking, quad safety,
    and energy penalties.
    """
    # verify obs dims before any slicing
    assert obs.shape[0] == self.OBS_SIZE, f"obs length {obs.shape[0]} != expected {self.OBS_SIZE}"

    # lambda for exponential reward
    er = lambda x, s=2: jp.exp(-s * jp.abs(x))

    # Team observations: payload error and linear velocity.

    pos_error = obs[:3]


    dis = jp.linalg.norm(pos_error)
    z_error = jp.abs(pos_error[2])
    distance_reward =  er(dis)#, 2 + sim_time)
    z_distance_reward =  er(z_error)#, 2 + sim_time)

    


    # Safety and smoothness penalties.
    quad1_obs = obs   

    collision_penalty = 1.0 * collision
    # out_of_bounds_penalty = 50.0 * out_of_bounds

    # Reward for quad orientations (encouraging them to remain upright).
    up_reward = er(angle_q1) 

    # # taut string reward
    # quad1_dist = jp.linalg.norm(quad1_obs[:3]) # payload to quad1
    # quad2_dist = jp.linalg.norm(quad2_obs[:3]) # payload to quad2
    # taut_reward = quad1_dist + quad2_dist # Maximize the string length
    # taut_reward += quad1_obs[2] + quad2_obs[2] # Maximize the height of the quads
    # taut_reward /= self.cable_length

    # Reward for quad velocities.
    # The reward is higher for lower angular velocities.
    # The reward is higher for lower linear velocities. 
    ang_vel_q1 = quad1_obs[15:18] 


    ang_vel_reward = er(jp.linalg.norm(ang_vel_q1), 20 * er(dis, 5)) # lower angvel range closer to target
    linvel_q1 = quad1_obs[12:15] 

    linvel_quad_reward =  er(jp.linalg.norm(linvel_q1),20 * er(dis,5)) # lower linvel range closer to target

    # Velocity alignment.
    target_dir  = pos_error / (dis + 1e-6)
    vel = jp.linalg.norm(linvel_q1)
    # Avoid division by zero. 
    vel_dir = jp.where(jp.abs(vel) > 1e-6, linvel_q1 / vel, jp.zeros_like(linvel_q1))
  

    aligned_vel = er(1 - jp.dot(vel_dir, target_dir), dis) # dotprod = 1  => vel is perfectly aligned
    velocity_towards_target = aligned_vel
  

    # vel_cap = 3.45 - 0.115 * vel**4
    # zero_at_target = 14.7 * dis * jp.exp(-10.5 * dis * jp.abs(vel))
    # no_zero_while_error = jp.exp(- (0.5 / (26.0 * dis + 0.3)) * jp.abs(vel))
    # target_reward = no_zero_while_error * (vel_cap - zero_at_target)
    # target_reward = jp.exp(0.4 * target_reward)
    # target_reward *= aligned_vel
    # target_reward *= jp.exp(-1.4 * jp.abs(dis))


    smooth_action_penalty = jp.mean(jp.abs(action - last_action))
    smooth_action_penalty /= self.time_per_action * 1000  # normlize for frequency

    action_energy_penalty = jp.mean((0.5 * (action + 1))**2)



    tracking_reward = self.reward_coeffs["distance_reward_coef"] * distance_reward
    #tracking_reward += self.reward_coeffs["z_distance_reward_coef"] * z_distance_reward
    tracking_reward += self.reward_coeffs["velocity_reward_coef"] * velocity_towards_target
    # tracking_reward += self.reward_coeffs.get("target_reward_coef", 1.0) * target_reward
    #tracking_reward = target_reward

    stability_reward = self.reward_coeffs["up_reward_coef"] * up_reward
    stability_reward += self.reward_coeffs["ang_vel_reward_coef"] * ang_vel_reward
    #stability_reward += self.reward_coeffs["linvel_reward_coef"] * linvel_reward
    stability_reward += self.reward_coeffs["linvel_quad_reward_coef"] * linvel_quad_reward
    #stability_reward += self.reward_coeffs["taut_reward_coef"] * taut_reward
    

    #penalties
    safety_reward = self.reward_coeffs["collision_penalty_coef"] * collision_penalty
    safety_reward += self.reward_coeffs["out_of_bounds_penalty_coef"] * out_of_bounds
    safety_reward += self.reward_coeffs["smooth_action_coef"] * smooth_action_penalty
    safety_reward += self.reward_coeffs["action_energy_coef"] * action_energy_penalty
    #safety_reward += self.reward_coeffs["safe_distance_coef"] * safe_distance_reward

  
    # Combine all rewards and penalties.
   
    reward = tracking_reward * (stability_reward + safety_reward)
    
  

    return reward, None, {}

# Register the environment under the name 'multiquad'
envs.register_environment('quad', QuadEnv)