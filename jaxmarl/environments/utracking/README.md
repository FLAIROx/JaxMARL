# Underwater Acoustic Tracking (UTracking)

**UTracking** is a multi-agent reinforcement learning (MARL) environment for **cooperative underwater acoustic tracking**. It models a fleet of autonomous underwater vehicles (AUVs) that must **track one or more moving underwater targets (landmarks)** using **range-only acoustic measurements**, intermittent communication, and noisy dynamics.

Policies trained in UTracking can be **directly deployed and evaluated** in Gazebo realistic simulations via **PyLrauv**.

## Environment Overview

![UTracking 3v3 Demo](/docs/imgs/utracking_3v3.gif)

### Goal
Learn decentralized policies that allow multiple agents to:
- Cooperatively follow and surround moving underwater targets
- Maintain acoustic contact despite noise, delays, and communication loss
- Avoid collisions while covering all targets

### Wht Utracking is difficult
- **Partial observability**: information acquired by agents is based on intermittent and noisy acoustic communications.
-	**Coordination**: successful tracking in challenging settings requires near-perfect coordination among agents.
-	**Stochastic dynamics**: agents’ trajectories are influenced by stochastic sea dynamics.
-	**Approximate information**: estimates of target positions are obtained through noisy Particle Filters (PF).
-	**Long-horizon missions**: tasks span hours of simulated time, requiring robust long-term decision making.


## RL Environment Specification

### Observation Space


Each agent receives a **local observation** describing other agents and landmarks.

| Feature | Description |
|------|------------|
| Δx, Δy, Δz | Relative position (self-motion or last communicated position) |
| range | Acoustic range (optional, by default is masked and agents only observe the Δ information based on current estimations of Particle Filters) |
| is_agent | Whether the entity is an agent |
| is_self | Whether the entity is the observing agent |

Additionally, a `world_state` is provided containing absolute information of the environment for each entity (agents and landmarks):

**Vertex State (default)**: 

| Feature | Description |
|------|------------|
| pos_x, pos_y, pos_z | Absolute position in space |
| direction | Heading angle in radians |
| vel | Entity velocity |
| pred_error | 2D tracking error for landmarks; 0 for agents |
| is_agent | 1 if agent, 0 if landmark |

**Edge State (if `state_as_edges=True`)**: 

| Feature | Description |
|------|------------|
| Δx, Δy, Δz | True relative position to other entity |
| obs_Δx, obs_Δy, obs_Δz | Observed relative position (from communication or tracking) |
| range | Acoustic range measurement |
| is_agent, is_landmark | One-hot encoding of entity type |

Notice that the observations and world_state will be returned as a matrix or flattened depending on parameters `matrix_obs` and `matrix_state` (set to `False` by default)
- **Matrix observation**: `(num_entities, number_of_feats)`
- **Flattened observation**: `(num_entities × number_of_feats)`

**Important details**
- Agent–agent relative positions are based on **last successfully communicated states**
- Landmark positions are **tracking predictions**, not ground truth
- Observations are normalized using `space_unit` (default: kilometers)

---

### Action Space

Agents control only the rudder only of the AUVs (the velocity is fixed). This means that actions correspond to changes in the heading of the vehicles. The relationship between rudder change and heading change are defined by linear models which coefficents can be found at [jaxmarl/environments/utracking/traj_models/traj_linear_models.json](traj_models/traj_linear_models.json)

- **Discrete (default)**: 5 rudder commands `[-0.24, -0.12, 0, 0.12, 0.24]`

- **Continuous (optional)**: continuous rudder angle in `[-0.24, 0.24]`

Note: Discrete actions are **locally constrained**: agents can only move to adjacent rudder angles. Prevents unrealistic sharp turns and improves sim-to-real transfer.

---

### Reward Function

The reward function balances **cooperative coverage** of targets with **tracking accuracy**, while penalizing unsafe behaviors. The total reward at each timestep is:

```
reward = rew_follow_coeff × follow_reward + rew_tracking_coeff × tracking_reward
```

If `rew_norm_landmarks=True` (default), the reward is normalized by dividing by the number of landmarks.

#### 1) Follow Reward (Coverage Component)

The follow reward counts how many landmarks have **at least one agent within tracking range**:

```python
follow_reward = count(landmarks with min_distance_to_any_agent ≤ rew_dist_thr)
```

**Purpose**: Encourages agents to distribute themselves across all targets rather than clustering around a single landmark.

**Range**: `[0, num_landmarks]` (or `[0, 1]` if normalized)

**Key Parameter**: `rew_dist_thr` (default: 180m) — the maximum distance at which a landmark is considered "followed"

#### 2) Tracking Reward (Accuracy Component)

The tracking reward evaluates the **quality of position estimates** for each landmark using an exponential decay function based on tracking error:

```python
tracking_reward = Σ exp_decay(tracking_error_i)
```

where the exponential decay function is:
- Returns **1.0** if error ≤ `rew_pred_ideal` (perfect tracking)
- Smoothly decays to **0.0** as error increases to `rew_pred_thr`
- Returns **0.0** if error ≥ `rew_pred_thr` (tracking too poor to reward)

The tracking error for each landmark is computed using the **best prediction** among all agents (i.e., the prediction from the agent with the smallest range measurement to that landmark).

**Purpose**: Rewards maintaining accurate position estimates, incentivizing agents to stay close enough for precise tracking.

**Range**: `[0, num_landmarks]` (or `[0, 1]` if normalized)

**Key Parameters**: 
- `rew_pred_ideal` (default: 10m) — ideal tracking error for full reward
- `rew_pred_thr` (default: 50m) — maximum acceptable tracking error
- `rew_tracking_coeff` (default: 0.0) — weight for tracking reward (disabled by default)

#### 3) Penalties

**Collision Penalty** (`penalty_for_crashing=True` by default):
- Sets reward to **−1.0** if any two agents are closer than `min_valid_distance` (default: 20m)
- Encourages safe navigation and collision avoidance

**Lost Agent Penalty** (`penalty_for_lost_agent=False` by default):
- Sets the reward to **0.0** for any step in which an agent has lost contact with all landmarks (closest landmark distance > `max_range_dist`)
- When enabled, creates a stricter incentive for maintaining connectivity (does not terminate the episode)


### Termination Conditions

An episode terminates only when the maximum number of steps (`max_steps`) is reached. Crashes and lost-agent events are handled through the reward (penalties / zeroed reward) rather than early termination, so all episodes have a fixed horizon.

---

## Hyperparameters

UTracking exposes numerous parameters to control **environment difficulty, sensing realism, and learning dynamics**. Parameters can be grouped into several categories for easier understanding.

### Environment Setup

Core parameters defining the mission structure and duration:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_agents` | int | 2 | Number of AUVs in the fleet |
| `num_landmarks` | int | 2 | Number of underwater targets to track |
| `dt` | int | 30 | Simulation timestep in seconds (must be in {20, 30, 60}) |
| `max_steps` | int | 256 | Maximum episode length in timesteps |
| `discrete_actions` | bool | True | Use discrete action space (5 rudder angles) vs continuous |

### Difficulty Presets

The `difficulty` parameter provides preset configurations for different challenge levels:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `difficulty` | str | "medium" | Preset difficulty: `"manual"`, `"easy"`, `"medium"`, `"hard"`, `"expert"` |

**Difficulty presets automatically configure** `landmark_rel_speed` (landmark speed as a fraction of agent speed) and `dirchange_time_range_landmark` (how many timesteps landmarks keep a heading before turning). Lower direction-change times mean more frequent, less predictable maneuvers:
- **easy**: Slow landmarks (`landmark_rel_speed=[0.1, 0.35]`), infrequent turns (`dirchange_time_range_landmark=[50, 200]`)
- **medium**: Moderate speed (`[0.2, 0.5]`) and turn frequency (`[35, 100]`)
- **hard**: Fast landmarks (`[0.5, 0.7]`) turning more often (`[10, 50]`)
- **expert**: Very fast targets (`[0.83, 0.86]`) with frequent, sharp maneuvers (`[5, 15]`)
- **manual**: Use individually specified parameters (ignores preset)

Note: presets do **not** modify `rudder_range_landmark` — the turn sharpness is taken from the explicitly provided value (default `[0.10, 0.25]`).

### Agent & Target Dynamics

Parameters controlling the movement capabilities and behavior of agents and landmarks:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prop_agent` | int | 30 | Agent propeller RPM (defines velocity; 30 RPM ≈ 1 m/s) |
| `landmark_rel_speed` | tuple | (0.1, 0.5) | Landmark speed as fraction of agent speed (min, max) |
| `rudder_range_landmark` | tuple | (0.10, 0.25) | Landmark rudder angle range in radians (controls turn sharpness) |
| `dirchange_time_range_landmark` | tuple | (5, 15) | Random timesteps between landmark direction changes (min, max) |
| `agent_depth` | tuple | (0.0, 0.0003) | Depth range for spawning agents (min, max in km) |
| `landmark_depth` | tuple | (5.0, 20.0) | Depth range for spawning landmarks (min, max in meters) |
| `landmark_depth_known` | bool | True | Whether agents know the depth of landmarks |

### Sensing & Communication

Parameters modeling the underwater acoustic environment:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_range_dist` | float | 1500.0 | Maximum range for acoustic sensing (meters); beyond this no landmark range is received |
| `max_range_for_prediction` | float | 600.0 | Ranges above this distance are considered unreliable and masked out of the tracking/prediction update |
| `max_comm_dist` | float | 1500.0 | Maximum range for agent-to-agent communication (meters) |
| `range_noise_std` | float | 10.0 | Standard deviation of Gaussian noise on range measurements (meters) |
| `agent_obs_noise_std` | float | 5.0 | Standard deviation of noise on communicated agent positions (meters) |
| `lost_comm_prob` | float | 0.1 | Probability of communication dropout per measurement |
| `steps_for_new_range` | int | 1 | Timesteps between range measurements (>1 reduces sensing frequency) |
| `steps_for_agent_communication` | int | 1 | Timesteps between agent communications (>1 reduces comm frequency) |

**Note**: Higher noise, communication loss, and sparser measurements increase realism but make the task harder.

### Tracking & State Estimation

Parameters for the landmark position estimation subsystem:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tracking_method` | str | "pf" | Method for estimating landmark positions: `"ls"` (Least Squares) or `"pf"` (Particle Filter) |
| `tracking_buffer_len` | int | 32 | Number of historical range observations stored for the LS method (ignored by PF, where it is internally set to `num_agents` and used only to hold the current step's inter-agent communication) |
| `min_steps_ls` | int | 2 | Minimum observations required before LS predictions start |
| `pf_num_particles` | int | 5000 | Number of particles used by the Particle Filter |

**Tracking method comparison**:
- **Least Squares (`"ls"`)**: Simpler, faster, works well with low noise. Requires `steps_for_new_range=1`.
- **Particle Filter (`"pf"`)**: More robust to noise and nonlinear dynamics, better for realistic scenarios.

### Noise & Stochasticity

Additional environmental noise sources for increased realism:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `traj_noise_std` | float | 0.05 | Standard deviation of noise on trajectory model (radians) |
| `velocity_noise_std` | float | 0.1 | Standard deviation of noise on agent velocities (m/s) |

### Rewards & Penalties

Parameters controlling the reward function structure:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rew_dist_thr` | float | 180.0 | Distance threshold for "following" a landmark (meters) |
| `rew_pred_ideal` | float | 10.0 | Ideal tracking error for maximum tracking reward (meters) |
| `rew_pred_thr` | float | 50.0 | Maximum tracking error threshold (meters) |
| `rew_norm_landmarks` | bool | True | Normalize rewards by number of landmarks |
| `rew_follow_coeff` | float | 1.0 | Weight coefficient for follow/coverage reward |
| `rew_tracking_coeff` | float | 0.0 | Weight coefficient for tracking accuracy reward |
| `penalty_for_crashing` | bool | True | Apply −1 reward for agent collisions |
| `penalty_for_lost_agent` | bool | False | Terminate episode if any agent loses all landmarks |

**Tuning guidance**:
- Increase `rew_tracking_coeff` to emphasize accurate state estimation over just coverage
- Set `penalty_for_lost_agent=True` for stricter connectivity requirements
- Adjust `rew_dist_thr` based on sensing capabilities and mission requirements

### Safety & Initialization

Parameters for collision avoidance and initial conditions:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_valid_distance` | float | 20.0 | Minimum safe distance between agents (collision threshold in meters) |
| `min_init_distance` | float | 30.0 | Minimum initial separation between any two entities (meters) |
| `max_init_distance` | float | 500.0 | Maximum initial separation between entities (meters) |
| `pre_init_pos` | bool | True | Precompute valid initial positions (faster resets) |
| `pre_init_pos_len` | int | 100000 | Number of initial configurations to precompute |
| `seed_init_pos` | int | 0 | Random seed for precomputing initial positions |

### Observation & State Representation

Parameters controlling how information is represented:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ranges_in_obs` | bool | False | Include raw range measurements in observations (default: only relative positions) |
| `matrix_obs` | bool | False | Return observations as matrices (entities × features) vs flattened vectors |
| `matrix_state` | bool | False | Return global state as matrix vs flattened |
| `state_as_edges` | bool | False | Represent state as edge features (pairwise) vs vertex features (per-entity) |
| `space_unit` | float | 1e-3 | Scaling factor for spatial observations (default: km, scaled to 0-1 range) |

### Utilities

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `infos_for_render` | bool | False | Include additional rendering information in info dict |

---

## Parameter Recommendations

### For Curriculum Learning
1. Start with `difficulty="easy"` and low noise (`range_noise_std=5.0`)
2. Gradually increase to `"medium"` → `"hard"` → `"expert"`
3. Progressively increase noise and communication dropout
4. Finally, reduce sensing frequency with `steps_for_new_range=2` or higher

### For Sim-to-Real Transfer
- Use `tracking_method="pf"` with realistic noise levels
- Set `range_noise_std=10-20`, `lost_comm_prob=0.1-0.2`
- Enable velocity and trajectory noise for robustness
- Consider `steps_for_new_range=4` to model realistic acoustic delays

### For Multi-Agent Coordination Research
- Increase `num_agents` and `num_landmarks` (e.g., 3v3, 5v5)
- Try to learn an optimal policy minimizing crashing and agents lost



## Current Limitations

UTracking implements **communication delays** through `steps_for_new_range` and `steps_for_agent_communication`, but in such a way that environment remains **fully synchronous**. These parameters are simplifications meant to approximate the kind of delays present in real acoustic systems.

**How the two delay parameters work**
- `steps_for_new_range` controls how often agents obtain a **new landmark range**. Between updates, the cached ranges and the frozen tracking predictions are reused; a new range is only computed every `steps_for_new_range` steps.
- `steps_for_agent_communication` controls how often agents refresh their **observation of other agents' positions**. Between updates, the last successfully communicated position is reused.

**What is simplified / assumed**
- **Fixed, not time-varying delays.** Both intervals are constant for the whole episode. In real deployments the effective delay varies during a mission (range-dependent acoustic travel time, congestion, packet loss bursts, etc.); this variability is not modeled.
- **No intra-signal lag for ranges.** When a new range arrives every `steps_for_new_range` steps (e.g. every 200 s), it is treated as a measurement taken *exactly* at the end of that interval, against the landmark's current position. There is no lag between when the acoustic signal was emitted and when it is consumed. 
- **Ranges are shared immediately.** Every time a new range is received it is **instantly broadcast to all other agents** (subject to the usual `lost_comm_prob` / `max_comm_dist` dropouts), regardless of `steps_for_agent_communication`. That parameter delays *only* the exchange of agent positions, not the sharing of landmark ranges.
- **No intra-signal lag for agent positions either.** As with ranges, a communicated agent position is assumed to be measured precisely at the end of the `steps_for_agent_communication` interval, even though a real delayed message would carry a stale position from when it was actually sent.

In short, even though some basic communication delays are present, the environment is **synchronous**: all quantities are computed from the agents' and landmarks' current positions, and "delayed" information is simply a fresh measurement made less frequently rather than a genuinely time-lagged one. 

**Next step:** a more robust **asynchronous** environment with **time-aware observation vectors** (observations tagged with the time at which each piece of information was actually acquired), so that policies can reason explicitly about the age of each range and communicated position.

## Connection with PyLrauv (Gazebo Evaluation)

<p align="center">
  <img width="40%" src="https://raw.githubusercontent.com/wiki/osrf/lrauv/media/LRUAV_3D.gif" alt="LRAUV 3D">
</p>


UTracking is designed to be **dynamically consistent** with the **Gazebo LRAUV simulator** via **PyLrauv**. This allows to test the trained policies in a more realistic simulation before deployment on real robots. 

**Links**
- PyLrauv repository: https://github.com/mttga/pylrauv/tree/main  
- JAX agent evaluation example: https://github.com/mttga/pylrauv/blob/main/lrauv_env/test_jax_agent.py
