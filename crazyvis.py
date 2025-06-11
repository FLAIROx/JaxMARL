import argparse
import numpy as np
import tensorflow as tf
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jaxmarl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_envs', type=int, default=128, help='Batch size')
    parser.add_argument('--timesteps', type=int, default=500, help='Rollout length')
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

    # 3) Reset N envs in parallel and stack their obs
    rng = jax.random.PRNGKey(0)
    pipeline_states = []
    obs_batch = []
    for i in range(args.num_envs):
        rng, key = jax.random.split(rng)
        obs, state = env.reset(key)            # jaxmarl API returns (obs, state)
        pipeline_states.append(state)
        obs_batch.append(np.array(obs))
    obs_batch = np.stack(obs_batch).astype(np.float32)

    # 4) Rollout for T timesteps
    for _ in range(args.timesteps):
        # run actor_model inference
        interpreter.set_tensor(inp_det['index'], obs_batch)
        interpreter.invoke()
        acts = interpreter.get_tensor(out_det['index'])  # shape [N, act_dim]

        # step each env individually
        next_ps = []
        next_obs = []
        for i in range(args.num_envs):
            rng, key = jax.random.split(rng)
            o2, s2, *_ = env.step(key, pipeline_states[i], jnp.array(acts[i]))
            next_ps.append(s2)
            next_obs.append(np.array(o2))
        pipeline_states = next_ps
        obs_batch = np.stack(next_obs).astype(np.float32)

    # 5) Compute true distances at final timestep and plot histogram
    distances = []
    for ps in pipeline_states:
        pos = np.array(ps.xpos[env.payload_body_id])       # payload position
        tgt = np.array(env.target_position)                # goal center
        distances.append(np.linalg.norm(tgt - pos))

    plt.hist(distances, bins=50)
    plt.xlabel('Payload–Target Distance')
    plt.ylabel('Frequency')
    plt.title(f'Distance Distribution at T={args.timesteps}')
    plt.savefig('distances_histogram.png')
    print('Saved distances_histogram.png')
    plt.show()

if __name__ == '__main__':
    main()
