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
import numpy as np

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
    mj_model = mujoco.MjModel.from_xml_path("mujoco/two_quad_payload.xml")
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
        sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, "q0_cf2")
    self.q2_body_id = mujoco.mj_name2id(
        sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, "q1_cf2")

    # (Optionally) Register any additional model components (e.g. goal marker).

  def reset(self, rng: jax.Array) -> State:
    """Resets the environment to an initial state."""
    rng, rng1, rng2 = jax.random.split(rng, 3)
    qpos = self.sys.qpos0 + jax.random.uniform(
        rng1, (self.sys.nq,), minval=-self._reset_noise_scale, maxval=self._reset_noise_scale)
    qvel = jax.random.uniform(
        rng2, (self.sys.nv,), minval=-self._reset_noise_scale, maxval=self._reset_noise_scale)
        
    rng, rng_euler = jax.random.split(rng, 2)
    keys = jax.random.split(rng_euler, 6)
    angle_range = 30 * jp.pi / 180  # 30 degrees in radians
    # Quad 1: sample roll, pitch, yaw.
    roll_q1 = jax.random.uniform(keys[0], minval=-angle_range, maxval=angle_range)
    pitch_q1 = jax.random.uniform(keys[1], minval=-angle_range, maxval=angle_range)
    yaw_q1 = jax.random.uniform(keys[2], minval=-jp.pi, maxval=jp.pi)
    # Quad 2: sample roll, pitch, yaw.
    roll_q2 = jax.random.uniform(keys[3], minval=-angle_range, maxval=angle_range)
    pitch_q2 = jax.random.uniform(keys[4], minval=-angle_range, maxval=angle_range)
    yaw_q2 = jax.random.uniform(keys[5], minval=-jp.pi, maxval=jp.pi)
    
    def euler_to_quat(roll, pitch, yaw):
        cr = jp.cos(roll * 0.5)
        sr = jp.sin(roll * 0.5)
        cp = jp.cos(pitch * 0.5)
        sp = jp.sin(pitch * 0.5)
        cy = jp.cos(yaw * 0.5)
        sy = jp.sin(yaw * 0.5)
        return jp.array([
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy
        ])
        
    quat_q1 = euler_to_quat(roll_q1, pitch_q1, yaw_q1)
    quat_q2 = euler_to_quat(roll_q2, pitch_q2, yaw_q2)
    quat_q1_corrected = jp.array([quat_q1[1], quat_q1[2], quat_q1[3], quat_q1[0]])
    quat_q2_corrected = jp.array([quat_q2[1], quat_q2[2], quat_q2[3], quat_q2[0]])
    start_q1 = self.q1_body_id * 7 + 3
    start_q2 = self.q2_body_id * 7 + 3
    qpos = qpos.at[start_q1:start_q1+4].set(quat_q1_corrected)
    qpos = qpos.at[start_q2:start_q2+4].set(quat_q2_corrected)
    
    pipeline_state = self.pipeline_init(qpos, qvel)
    # Initialize last action as zeros.
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

    # local-frame observations for each quad.
    # For quad1:
    R1 = jp_R_from_quat(quad1_quat)      # rotation matrix: local -> global
    R1_T = jp.transpose(R1)              # global -> local
    local_quad1_rel         = jp.matmul(R1_T, quad1_rel)
    local_quad1_linvel      = jp.matmul(R1_T, quad1_linvel)
    local_quad1_angvel      = jp.matmul(R1_T, quad1_angvel)
    local_quad1_linear_acc  = jp.matmul(R1_T, quad1_linear_acc)
    local_quad1_angular_acc = jp.matmul(R1_T, quad1_angular_acc)
    q1_q2_rel = quad2_pos - quad1_pos
    local_q1_q2_rel = jp.matmul(R1_T, q1_q2_rel)
    local_q1_payload_error = jp.matmul(R1_T, payload_error)
    local_q1_payload_linvel = jp.matmul(R1_T, payload_linvel)

    # For quad2:
    R2 = jp_R_from_quat(quad2_quat)
    R2_T = jp.transpose(R2)
    local_quad2_rel         = jp.matmul(R2_T, quad2_rel)
    local_quad2_linvel      = jp.matmul(R2_T, quad2_linvel)
    local_quad2_angvel      = jp.matmul(R2_T, quad2_angvel)
    local_quad2_linear_acc  = jp.matmul(R2_T, quad2_linear_acc)
    local_quad2_angular_acc = jp.matmul(R2_T, quad2_angular_acc)
    local_q2_q1_rel = jp.matmul(R2_T, -q1_q2_rel)
    local_q2_payload_error = jp.matmul(R2_T, payload_error)
    local_q2_payload_linvel = jp.matmul(R2_T, payload_linvel)

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
        local_quad1_rel,      # (3,)  62-64
        local_quad1_linvel,   # (3,)  65-67
        local_quad1_angvel,   # (3,)  68-70
        local_quad1_linear_acc, # (3,) 71-73
        local_quad1_angular_acc, # (3,) 74-76
        local_quad2_rel,      # (3,)  77-79
        local_quad2_linvel,   # (3,)  80-82
        local_quad2_angvel,   # (3,)  83-85
        local_quad2_linear_acc, # (3,) 86-88
        local_quad2_angular_acc, # (3,) 89-91
        local_q1_q2_rel,      # (3,)  92-94
        local_q2_q1_rel,      # (3,)  95-97
        local_q1_payload_error, # (3,) 98-100
        local_q2_payload_error, # (3,) 101-103
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