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
from .quad_env_builder import QuadEnvGenerator

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
      reward_coeffs: dict = None,
      obs_noise: float = 0.0,           # Parameter for observation noise
      act_noise: float = 0.0,         # Parameter for actuator noise
      max_thrust_range: float = 0.3,               # range for randomizing thrust
      num_quads = 2,
      cable_length: float = 0.4,  # Length of the cable connecting the payload to the quadrotors.
      trajectory = None,  # array of target positions for the payload
      target_start_ratio: float = 0.2,  # percentage of resets to target position
      payload_mass: float = 0.01,  # Mass of the payload.
      **kwargs,
  ):
    print("Initializing MultiQuadEnv")
    # Dynamically generate the MuJoCo XML via QuadEnvGenerator
    gen = QuadEnvGenerator(n_quads=num_quads, cable_length=cable_length, payload_mass=payload_mass)
    xml = gen.generate_xml()
    mj_model = mujoco.MjModel.from_xml_string(xml)
    # Convert the MuJoCo model to a Brax system.
    sys = mjcf.load_model(mj_model)
    kwargs['n_frames'] = kwargs.get('n_frames', sim_steps_per_action)
    kwargs['backend'] = 'mjx'
    super().__init__(sys, **kwargs)

    self.cable_length = cable_length
    self.target_start_ratio = target_start_ratio

    # Save environment parameters.
    self.policy_freq = policy_freq
    self.sim_steps_per_action = sim_steps_per_action
    self.time_per_action = 1.0 / self.policy_freq
  

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
    self.trajectory = trajectory
    if  self.trajectory is not None:
      #make sure the trajectory has length of episode_length
      self.trajectory = jp.array(trajectory, dtype=jp.float32)
      if self.trajectory.ndim == 1:
        # If trajectory is a 1D array, reshape it to (episode_length, 3).
        self.trajectory = self.trajectory.reshape(-1, 3)
      elif self.trajectory.ndim == 2 and self.trajectory.shape[1] != 3:
        # If trajectory is a 2D array but not of shape (episode_length, 3), raise an error.
        raise ValueError("Trajectory must be of shape (episode_length, 3) or (episode_length * 3,).")

      # If a trajectory is provided, set the target position to the first point.
      self.target_position = jp.array(trajectory[0], dtype=jp.float32)


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

    self.obs_table = obs_table

  @staticmethod
  def generate_configuration(key, num_quads, cable_length, target_position, target_start_ratio):
    # split once for payload and quads
    subkeys = jax.random.split(key, 5)
    min_qz, min_pz = 0.008, 0.0055

    # payload
    xy = jax.random.uniform(subkeys[0], (2,), minval=-1.5, maxval=1.5)
    pz = jax.random.uniform(subkeys[1], (), minval=-1.0, maxval=3.0)
    payload = jp.array([xy[0], xy[1], pz.clip(min_pz, 3.0)])

    # reset payload position to the target position with a probability
    is_target_start = jax.random.uniform(subkeys[1]) < target_start_ratio
    payload = jp.where(is_target_start, target_position + jax.random.normal(subkeys[1], (3,)) * 0.02, payload)

    # spherical params
    mean_r, std_r = cable_length, cable_length/3
    clip_r = (0.05, cable_length)
    mean_th, std_th = jp.pi/4, jp.pi/8
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
  def generate_filtered_configuration_batch(key, batch_size, num_quads, cable_length, target_position, target_start_ratio):
    # 1) oversample_factor=2
    os_factor = round(num_quads / 2) + 1
    M = os_factor * batch_size
    keys = jax.random.split(key, M)
    payloads, quadss = jax.vmap(
      MultiQuadEnv.generate_configuration, in_axes=(0, None, None, None, None)
    )(keys, num_quads, cable_length, target_position, target_start_ratio)  # shapes (M,3), (M,num_quads,3)

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
    # Randomize velocities around zero
    ang_vel_std = 20 * jp.pi / 180  # 20 degrees per second
    lin_vel_std = 0.2  # 20 cm/s

    qvel = jp.zeros(self.sys.nv)
    for i in range(self.num_quads):
      quad_body_id = self.quad_body_ids[i]
      lin = jax.random.normal(rng2, (3,)) * lin_vel_std
      ang = jax.random.normal(rng2, (3,)) * ang_vel_std
      start = quad_body_id * 6
      qvel = qvel.at[start:start+3].set(lin)
      qvel = qvel.at[start+3:start+6].set(ang)

    # Get new positions for payload and both quadrotors.
    payload_pos, quad_positions = MultiQuadEnv.generate_filtered_configuration_batch(
      rng_config, 1, self.num_quads, self.cable_length, self.target_position, self.target_start_ratio
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
    thrust_cmds = jp.clip(thrust_cmds, 0.0, 1.0)
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

    up = jp.array([0.0, 0.0, 1.0])
    # collect orientations & positions
    angles = []
    quad_positions = []
    for qb in self.quad_body_ids:
        quat = pipeline_state.xquat[qb]
        local_up = jp_R_from_quat(quat)[:, 2]
        angles.append(jp_angle_between(local_up, up))
        quad_positions.append(pipeline_state.xpos[qb])
    angles = jp.stack(angles)                           # (num_quads,)
    qp = jp.stack(quad_positions)                       # (num_quads,3)

    # pairwise quad-quad collision TODO: make this use proper mjx collision detection
    dists = jp.linalg.norm(qp[:, None, :] - qp[None, :, :], axis=-1)
    eye  = jp.eye(self.num_quads, dtype=bool)
    min_dist = jp.min(jp.where(eye, jp.inf, dists))
    quad_collision = min_dist < 0.15

    # ground collision if any quads AND payload near ground
    ground_collision_quad    = jp.any(qp[:, 2] < 0.03)
    ground_collision_payload = pipeline_state.xpos[self.payload_body_id][2] < 0.03
    ground_collision = jp.logical_or(ground_collision_quad, ground_collision_payload)
    collision       = jp.logical_or(quad_collision, ground_collision)

    # out-of-bounds if any quad tilts too far or goes under payload
    too_tilted = jp.any(jp.abs(angles) > jp.radians(150))
    below_pl   = jp.any(qp[:, 2] < pipeline_state.xpos[self.payload_body_id][2] - 0.15)
    out_of_bounds = jp.logical_or(too_tilted, below_pl)

    # out of bounds for pos error shrinking with time
    # payload_pos = pipeline_state.xpos[self.payload_body_id]
    # payload_error = self.target_position - payload_pos
    # payload_error_norm = jp.linalg.norm(payload_error)
    # max_time_to_target = self.max_time * 0.75
    # time_progress = jp.clip(pipeline_state.time / max_time_to_target, 0.0, 1.0)
    # max_payload_error = 4 * (1 - time_progress) + 0.05 # allow for 5cm error at the target
    # out_of_bounds = jp.logical_or(out_of_bounds, payload_error_norm > max_payload_error)


    # set target if trajectory is provided
    target_position = self.target_position
    if self.trajectory is not None and self.trajectory.shape[0] > 0:
      # get the next target position from the trajectory
      target_idx = jp.clip(
        jp.floor(pipeline_state.time  / self.time_per_action).astype(jp.int32),
        0, self.trajectory.shape[0] - 1
      ) 
      target_position = self.trajectory[target_idx]


    obs = self._get_obs(pipeline_state, action, target_position, noise_key)
    reward, _, _ = self.calc_reward(
        obs, pipeline_state.time, collision, out_of_bounds,
        action, angles, prev_last_action,
        target_position, pipeline_state, max_thrust
    )

    # dont terminate ground collision on ground start
    ground_collision = jp.logical_and(
      ground_collision,
      jp.logical_or(
        pipeline_state.time > 3, # allow 2 seconds for takeoff
        pipeline_state.cvel[self.payload_body_id][2] < -3.0,
      )
    )

    collision = jp.logical_or(quad_collision, ground_collision)
    
    done = jp.logical_or(out_of_bounds, collision)
   
    
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
    obs_list.append(jp.clip(last_action, -1.0, 1.0))
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

  def calc_reward(self, obs, sim_time, collision, out_of_bounds,
                  action, angles, last_action, target_position, data,
                  max_thrust) -> (jp.ndarray, None, dict):
    
    er = lambda x, s=2: jp.exp(-s * jp.abs(x))

    # payload tracking rewards
    team_obs      = obs[:6]
    payload_err   = team_obs[:3]
    payload_linlv = team_obs[3:6]
    dis = jp.linalg.norm(payload_err)
    tracking_reward = self.reward_coeffs["distance_reward_coef"] * er(dis)

    
    quad_obs = [obs[6 + i*24 : 6 + (i+1)*24] for i in range(self.num_quads)]
    rels     = jp.stack([q[:3]     for q in quad_obs])  # (num_quads,3)
    rots     = jp.stack([q[3:12]  for q in quad_obs])  # (num_quads,9)
    linvels  = jp.stack([q[12:15] for q in quad_obs])  # (num_quads,3)
    angvels  = jp.stack([q[15:18] for q in quad_obs])  # (num_quads,3)
    

    # safe-distance reward (mean over all pairs)
    if self.num_quads > 1:
        d = jp.linalg.norm(rels[:, None, :] - rels[None, :, :], axis=-1)
        eye = jp.eye(self.num_quads, dtype=bool)
        pairwise = jp.where(eye, jp.inf, d)
        safe_distance = jp.mean(jp.clip(3*(pairwise - 0.18) / 0.02, -20, 1))

    else:
        safe_distance = 1.0

    # upright reward = mean over all quads
    up_reward = jp.mean(er(angles))

    # taut-string reward = sum of distances + heights
    quad_dists   = jp.linalg.norm(rels, axis=-1)
    quad_heights = rels[:, 2]
    taut_reward  = (jp.sum(quad_dists) + jp.sum(quad_heights)) / self.cable_length

    # angular and linear velocity rewards summed
    ang_vel_vals = jp.stack([er(jp.linalg.norm(jvp, axis=-1)) for jvp in angvels])
    ang_vel_reward = (0.5 + 3 * er(dis, 20)) * jp.mean(ang_vel_vals)
    linvel_vals = jp.stack([er(jp.linalg.norm(jvp, axis=-1)) for jvp in linvels])
    linvel_quad_reward = (0.5 + 6 * er(dis, 20)) * jp.mean(linvel_vals)

    # penalties
    collision_penalty = self.reward_coeffs["collision_penalty_coef"] * collision
    oob_penalty       = self.reward_coeffs["out_of_bounds_penalty_coef"] * out_of_bounds

    smooth_penalty    = self.reward_coeffs["smooth_action_coef"] * jp.mean(jp.abs(action - last_action))
    thrust_cmds = 0.5 * (action + 1.0)
    thrust_extremes = jp.exp(-50 * jp.abs(thrust_cmds)) + jp.exp(50 * (thrust_cmds - 1)) # 1 if thrust_cmds is 0 or 1 and going to 0 in the middle
    # if actions out of bounds lead them to action space
    thrust_extremes = jp.where(jp.abs(action)> 1.0, 1.0 + 0.1*jp.abs(action), thrust_extremes)  

    energy_penalty    = self.reward_coeffs["action_energy_coef"] * jp.mean(thrust_extremes)


    stability = (self.reward_coeffs["up_reward_coef"] * up_reward
                 + self.reward_coeffs["taut_reward_coef"] * taut_reward
                 + self.reward_coeffs["ang_vel_reward_coef"] * ang_vel_reward
                 + self.reward_coeffs["linvel_quad_reward_coef"] * linvel_quad_reward)
    safety = safe_distance * self.reward_coeffs["safe_distance_coef"] \
           + collision_penalty + oob_penalty + smooth_penalty + energy_penalty

    reward = tracking_reward * (stability + safety)
  
    return reward, None, {}

# Register the environment under the name 'multiquad'
envs.register_environment('multiquad', MultiQuadEnv)