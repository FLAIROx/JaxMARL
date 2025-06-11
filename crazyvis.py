import argparse
import numpy as np
import tensorflow as tf
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jaxmarl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_envs', type=int, default=64,  help='Batch size')
    parser.add_argument('--timesteps', type=int, default=2000, help='Rollout length')
    parser.add_argument('--model_path', type=str, default='actor_model.tflite')
    args = parser.parse_args()

    # 1) Load TFLite model and resize input for batch inference
    interpreter = tf.lite.Interpreter(model_path=args.model_path)
    interpreter.allocate_tensors()
    inp_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    interpreter.resize_tensor_input(inp_det['index'], [args.num_envs, inp_det['shape'][-1]])
    interpreter.allocate_tensors()

    # 2) Create the environment (use jaxmarl.make for correct wrappers)
    env = jaxmarl.make(
      "multiquad_ix4",
      policy_freq=500,
      sim_steps_per_action=1,
      episode_length=args.timesteps,
      reward_coeffs=None,
      obs_noise=0.3,
      act_noise=0.1,
      max_thrust_range=0.3,
      num_quads=2,
      cable_length=0.4,
    )

    # 3) Run rollouts and collect final position errors
    rng = jax.random.PRNGKey(0)
    rng, reset_key = jax.random.split(rng)
    # reset returns batched obs and pipeline_state
    obs, state = env.reset(reset_key)
    for _ in range(args.timesteps):
        # run actor tflite inference
        model_input = np.array(obs, dtype=np.float32)
        interpreter.set_tensor(inp_det['index'], model_input)
        interpreter.invoke()
        action = interpreter.get_tensor(out_det['index'])
        # step the vectorized env
        rng, step_key = jax.random.split(rng)
        obs, state, _, _, _ = env.step_env(step_key, state, action)

    # compute per-env payload position error (first 3 dims of obs)
    pos_errs = np.linalg.norm(np.array(obs)[:, :3], axis=1)
    # plot histogram
    plt.figure()
    plt.hist(pos_errs, bins=50)
    plt.xlabel('Position Error')
    plt.ylabel('Frequency')
    plt.title('Position Error Histogram at Last Frame')
    plt.savefig('pos_err_histogram.png')
    print("Saved histogram to pos_err_histogram.png")
