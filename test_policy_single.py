import numpy as np
import jax
import jax.random
import time
import jaxmarl
import cv2
import tensorflow as tf

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
    "ENV_NAME": "quad_1x4",      
    "ENV_KWARGS": {
        "reward_coeffs": default_reward_coeffs,
        "obs_noise": 0.0,
        "act_noise": 0.0,
    },
}
env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
obs_shape = env.observation_spaces[env.agents[0]].shape[0]
act_dim = env.action_spaces[env.agents[0]].shape[0]

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="actor_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize environment state
rng = jax.random.PRNGKey(int(time.time()))
rng, reset_key = jax.random.split(rng)
state = env.reset(reset_key)[1]

# before entering the loop
last_render = time.time()
render_interval = 1.0 / 25.0

# Interactive simulation loop
while True:
    # Get observation for the first agent
    obs = env.get_obs(state)
    obs_agent = obs[env.agents[0]]  # shape: (obs_shape,)
    
    # Enforce exact match to model input size
    input_dim = input_details[0]['shape'][1]
    if obs_agent.size != input_dim:
        raise ValueError(f"Observation length {obs_agent.size} != model input {input_dim}")

    # Run TFLite inference
    model_input = obs_agent.astype(np.float32).reshape(1, input_dim)
    interpreter.set_tensor(input_details[0]['index'], model_input)
    interpreter.invoke()
   
   
    action = interpreter.get_tensor(output_details[0]['index'])[0]
    
    # Create actions dict for all agents (non-controlled agents get zero action)
    actions = {env.agents[0]: action}
    for agent in env.agents:
        if agent not in actions:
            actions[agent] = np.zeros(act_dim, dtype=np.float32)
    
    # Step environment
    rng, step_key = jax.random.split(rng)
    _, new_state, rewards, dones, info = env.step_env(step_key, state, actions)
    state = new_state

    # throttle rendering to 25 fps
    curr_time = time.time()
    if curr_time - last_render >= render_interval:
        frame = env.render([state], camera="track", width=640, height=480)[0]
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("Interactive Policy", frame_bgr)
        last_render = curr_time

    # always poll for quit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
