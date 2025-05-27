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
      num_quads = 2,
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

    # Cache body IDs

    # add number-of-quads from env var (default=2)
    self.num_quads = num_quads

    # cache quad body/joint/qpos-start indices in lists
    # assumes model has bodies "q0_container", ..., joints "q0_joint", ...
    self.quad_body_ids = [
      mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, f"q{i}_container")
      for i in range(self.num_quads)
    ]
    self.quad_joint_ids = [
      mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_JOINT.value, f"q{i}_joint")
      for i in range(self.num_quads)
    ]
    self.quad_qpos_starts = [sys.mj_model.jnt_qposadr[j] for j in self.quad_joint_ids]

    self.payload_body_id = mujoco.mj_name2id(
        sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, "payload")
    self.payload_joint_id = mujoco.mj_name2id(
        sys.mj_model, mujoco.mjtObj.mjOBJ_JOINT.value, "payload_joint")
    self.payload_qpos_start = sys.mj_model.jnt_qposadr[self.payload_joint_id]

    # --- ADD THESE HELPERS SO q1_/q2_ attrs exist ---
    self.q1_body_id     = self.quad_body_ids[0]
    self.q2_body_id     = self.quad_body_ids[1] if self.num_quads > 1 else self.quad_body_ids[0]
    self.q1_qpos_start  = self.quad_qpos_starts[0]
    self.q2_qpos_start  = self.quad_qpos_starts[1] if self.num_quads > 1 else self.quad_qpos_starts[0]
    # --- end helpers ---

    print("IDs:")
    print("Payload body ID:", self.payload_body_id)
    print("Payload joint ID:", self.payload_joint_id)
    print("Payload qpos start:", self.payload_qpos_start)

    print("Quadrotor body IDs and joint IDs:")
    for i in range(self.num_quads):
      print(f"Quad {i} body ID:", self.quad_body_ids[i])
      print(f"Quad {i} joint ID:", self.quad_joint_ids[i])
      print(f"Quad {i} qpos start:", self.quad_qpos_starts[i])



    print("Noise_level:", self.obs_noise)
    print("MultiQuadEnv initialized successfully.")

    # --- dynamic obs/action index table printout ---
    obs_table = []
    idx = 0
    obs_table.append((f"{idx}-{idx+2}", "payload_error")); idx += 3
    obs_table.append((f"{idx}-{idx+2}", "payload_linvel")); idx += 3
    for i in range(self.num_quads):
      for name, length in [
        ("rel", 3), ("rot", 9),
        ("linvel", 3), ("angvel", 3),
        ("linear_acc", 3), ("angular_acc", 3)
      ]:
        obs_table.append((f"{idx}-{idx+length-1}", f"quad{i}_{name}"))
        idx += length
    start = idx
    end = start + self.sys.nu - 1
    obs_table.append((f"{start}-{end}", "last_action"))

    print("Observation indices:")
    for r, nm in obs_table:
      print(f"  {r}: {nm}")
    # --- end table ---

  @staticmethod
  def generate_configuration(key, num_quads, cable_length):
    # split once for payload and quads
    subkeys = jax.random.split(key, 5)
    min_qz, min_pz = 0.008, 0.0055

    # payload
    xy = jax.random.uniform(subkeys[0], (2,), minval=-1.5, maxval=1.5)
    pz = jax.random.uniform(subkeys[1], (), minval=-1.0, maxval=3.0)
    payload = jp.array([xy[0], xy[1], pz.clip(min_pz, 3.0)])

    # spherical params
    mean_r, std_r = cable_length, cable_length/3
    clip_r = (0.05, cable_length)
    mean_th, std_th = jp.pi/7, jp.pi/8
    std_phi = jp.pi/(num_quads+1)
    # sample per-quad
    r     = jp.clip(mean_r + std_r*jax.random.normal(subkeys[2], (num_quads,)), *clip_r)
    th    = mean_th + std_th*jax.random.normal(subkeys[3], (num_quads,))
    offset= jax.random.uniform(subkeys[4], (), minval=-jp.pi, maxval=jp.pi)
    phi   = jp.arange(num_quads)*(2*jp.pi/num_quads) + std_phi*jax.random.normal(subkeys[4], (num_quads,)) + offset

    # to Cartesian
    x = r*jp.sin(th)*jp.cos(phi) + payload[0]
    y = r*jp.sin(th)*jp.sin(phi) + payload[1]
    z = jp.clip(r*jp.cos(th) + payload[2], min_qz, 3.0)
    quads = jp.stack([x, y, z], axis=1)

    return payload, quads


  @staticmethod
  def generate_filtered_configuration_batch(key, batch_size, num_quads, cable_length):
    # 1) oversample_factor=2
    M = 2 * batch_size
    keys = jax.random.split(key, M)
    payloads, quadss = jax.vmap(
      MultiQuadEnv.generate_configuration, in_axes=(0, None, None)
    )(keys, num_quads, cable_length)  # shapes (M,3), (M,num_quads,3)

    # 2) quad‐to‐quad min‐dist
    diffs = quadss[:, :, None, :] - quadss[:, None, :, :]
    dists = jp.linalg.norm(diffs, axis=-1)
    eye = jp.eye(num_quads, dtype=bool)[None]
    dists = jp.where(eye, jp.inf, dists)
    min_quad = jp.min(dists, axis=(1,2))

    # 3) quad‐to‐payload
    pd = quadss - payloads[:, None, :]
    min_payload = jp.min(jp.linalg.norm(pd, axis=-1), axis=1)

    # 4) mask & pick top batch_size
    mask = (min_quad>=0.16)&(min_payload>=0.07)
    idx = jp.argsort(-mask.astype(jp.int32))[:batch_size]

    return payloads[idx], quadss[idx]


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
    payload_pos, quad_positions = MultiQuadEnv.generate_filtered_configuration_batch(
      rng_config, 1, self.num_quads, self.cable_length
    )
    payload_pos = payload_pos[0]
    quad_positions = quad_positions[0]  # shape (num_quads, 3)

    # Generate orientations per quad and convert to quaternions
    rng, rng_euler = jax.random.split(rng, 2)
    keys = jax.random.split(rng_euler, self.num_quads * 3)
    std_dev = 10 * jp.pi / 180
    clip_val = 60 * jp.pi / 180
    quats = []
    def euler_to_quat(roll, pitch, yaw):
      cr = jp.cos(roll * 0.5); sr = jp.sin(roll * 0.5)
      cp = jp.cos(pitch * 0.5); sp = jp.sin(pitch * 0.5)
      cy = jp.cos(yaw * 0.5); sy = jp.sin(yaw * 0.5)
      return jp.array([cr*cp*cy+sr*sp*sy,
                       sr*cp*cy-cr*sp*sy,
                       cr*sp*cy+sr*cp*sy,
                       cr*cp*sy-sr*sp*cy])
    for i in range(self.num_quads):
      k0, k1, k2 = keys[3*i], keys[3*i+1], keys[3*i+2]
      roll  = jp.clip(jax.random.normal(k0) * std_dev, -clip_val, clip_val)
      pitch = jp.clip(jax.random.normal(k1) * std_dev, -clip_val, clip_val)
      yaw   = jax.random.uniform(k2, minval=-jp.pi, maxval=jp.pi)
      # zero out if too close to ground
      roll  = jp.where(quad_positions[i][2] < 0.02, 0.0, roll)
      pitch = jp.where(quad_positions[i][2] < 0.02, 0.0, pitch)
      quats.append(euler_to_quat(roll, pitch, yaw))
    quats = jp.stack(quats)

    # Build the full qpos vector dynamically
    new_qpos = base_qpos
    new_qpos = new_qpos.at[self.payload_qpos_start:
                           self.payload_qpos_start+3].set(payload_pos)
    for i in range(self.num_quads):
      start = self.quad_qpos_starts[i]
      new_qpos = new_qpos.at[start:start+3].set(quad_positions[i])
      new_qpos = new_qpos.at[start+3:start+7].set(quats[i])
    
    
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


    # build dynamic obs list
    obs_list = [payload_error, payload_linvel]
    for i in range(self.num_quads):
      pos = data.xpos[self.quad_body_ids[i]]
      quat = data.xquat[self.quad_body_ids[i]]
      rel = pos - payload_pos
      rot = jp_R_from_quat(quat).ravel()
      linvel = data.cvel[self.quad_body_ids[i]][3:6]
      angvel = data.cvel[self.quad_body_ids[i]][:3]
      linear_acc = data.cacc[self.quad_body_ids[i]][3:6]
      angular_acc = data.cacc[self.quad_body_ids[i]][:3]
      obs_list += [rel, rot, linvel, angvel, linear_acc, angular_acc]
    obs_list.append(last_action)
    obs = jp.concatenate(obs_list)

    # build dynamic noise lookup
    noise_list = [
      jp.ones(3)*0.005,  # payload_error
      jp.ones(3)*0.05    # payload_linvel
    ]
    for _ in range(self.num_quads):
      noise_list += [
        jp.ones(3)*0.02,    # rel
        jp.ones(9)*0.005,   # rot
        jp.ones(3)*0.05,    # linvel
        jp.ones(3)*0.08,    # angvel
        jp.ones(3)*0.05,    # linear_acc
        jp.ones(3)*0.05     # angular_acc
      ]
    noise_list.append(jp.ones(self.sys.nu)*0.0)  # last_action
    noise_lookup = jp.concatenate(noise_list)

    if self.obs_noise != 0.0:
      noise = self.obs_noise * noise_lookup * jax.random.normal(noise_key, obs.shape)
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

# Register the environment under the name 'multiquad'
envs.register_environment('multiquad', MultiQuadEnv)