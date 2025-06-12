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
    inp_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    # resize BEFORE allocate so batch dim is correct
    interpreter.resize_tensor_input(inp_det['index'],
                                    [args.num_envs, inp_det['shape'][-1]])
    interpreter.allocate_tensors()
    # get expected input dimension for tflite model
    input_dim = inp_det['shape'][-1]

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
    # grab agent names for splitting actions
    agent_names = env.agents

    # 3) Run rollouts in parallel and collect final position errors
    rng = jax.random.PRNGKey(0)
    # batch-reset
    subkeys = jax.random.split(rng, args.num_envs + 1)
    rng = subkeys[0]
    reset_keys = subkeys[1:]
    obs_batch_dict, states = jax.vmap(env.reset)(reset_keys)
    obs = np.array(obs_batch_dict['global'], dtype=np.float32)
    # slice observation to match model input dimension
    obs = obs[:, :input_dim]

    for _ in range(args.timesteps):
        # TFLite inference on batched obs
        interpreter.set_tensor(inp_det['index'], obs)
        interpreter.invoke()
        action = interpreter.get_tensor(out_det['index'])  # (batch, total_act_dim)

        # split or duplicate flat action into per-agent dict
        action_jax = jnp.array(action)
        total_dim = action_jax.shape[-1]
        # if model outputs per-agent action, duplicate across agents
        global_act_size = env.env.action_size
        if total_dim * len(agent_names) == global_act_size:
            actions = {agent: action_jax for agent in agent_names}
        else:
            per_dim = total_dim // len(agent_names)
            actions = {
                agent_names[i]: action_jax[:, i*per_dim:(i+1)*per_dim]
                for i in range(len(agent_names))
            }

        # generate new PRNG keys for stepping
        subkeys = jax.random.split(rng, args.num_envs + 1)
        rng = subkeys[0]
        step_keys = subkeys[1:]
        # batch-step using a pytree vmap over the dict
        obs_batch_dict, states, _, _, _ = jax.vmap(
            env.step_env,
            in_axes=(0, 0, {agent: 0 for agent in agent_names})
        )(step_keys, states, actions)
        obs = np.array(obs_batch_dict['global'], dtype=np.float32)
        # slice observation to match model input dimension
        obs = obs[:, :input_dim]
    
    # compute per-env payload position error from the first 3 dims
    pos_errs = np.linalg.norm(obs[:, :3], axis=1)
    # plot histogram
    plt.figure()
    plt.hist(pos_errs, bins=50)
    plt.xlabel('Position Error')
    plt.ylabel('Frequency')
    plt.title('Position Error Histogram at Last Frame')
    plt.savefig('pos_err_histogram.png')
    print("Saved histogram to pos_err_histogram.png")


if __name__ == "__main__":
    main()
    # Run the main function