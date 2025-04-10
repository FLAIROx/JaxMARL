import onnxruntime
import cv2
import numpy as np
import jax
import jax.random
import time
import jaxmarl

# Minimal configuration and environment creation (adjust if needed)
default_reward_coeffs = {
    "distance_reward_coef": 1.0,
    "z_distance_reward_coef": 0.0,
    "velocity_reward_coef": 1.0,
    "safe_distance_coef": 1.0,
    "up_reward_coef": 1.0,
    "linvel_reward_coef": 1.0,
    "ang_vel_reward_coef": 1.0,
    "linvel_quad_reward_coef": 1.0,
    "taut_reward_coef": 1.0,
    "collision_penalty_coef": -1.0,
    "out_of_bounds_penalty_coef": -1.0,
    "smooth_action_coef": -2.0,
    "action_energy_coef": 0.0,
}
config = {
    "ENV_NAME": "multiquad_2x4",
    "ENV_KWARGS": {
        "reward_coeffs": default_reward_coeffs,
        "obs_noise": 0.0,
        "act_noise": 0.0,
    },
}
env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
obs_shape = env.observation_spaces[env.agents[0]].shape[0]
act_dim = env.action_spaces[env.agents[0]].shape[0]

# Load the actor ONNX model
session = onnxruntime.InferenceSession("actor_model.onnx")
input_name = session.get_inputs()[0].name

# Initialize environment state
rng = jax.random.PRNGKey( time.time())
rng, reset_key = jax.random.split(rng)
state = env.reset(reset_key)[1]  # using state[1] as in training

# Interactive simulation loop
while True:
    # Get observation for the first agent
    obs = env.get_obs(state)
    obs_agent = obs[env.agents[0]]  # shape: (obs_shape,)
    model_input = np.array(obs_agent, dtype=np.float32).reshape(1, -1)
    
    # Run actor model inference and extract action
    outputs = session.run(None, {input_name: model_input})
    action = outputs[0][0]  # shape: (act_dim,)
    
    # Create actions dict for all agents (non-controlled agents get zero action)
    actions = {env.agents[0]: action}
    for agent in env.agents:
        if agent not in actions:
            actions[agent] = np.zeros(act_dim, dtype=np.float32)
    
    # Step environment
    rng, step_key = jax.random.split(rng)
    _, new_state, rewards, dones, info = env.step_env(step_key, state, actions)
    state = new_state
    
    # Render current state (env.render expects a list of states)
    frame = env.render([state], camera="track", width=640, height=480)[0]
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow("Interactive Policy", frame_bgr)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
