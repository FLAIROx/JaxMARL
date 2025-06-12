import argparse
import numpy as np
import tensorflow as tf
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jaxmarl
import asdf
from jax.tree_util import tree_map

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_envs', type=int, default=64, help='Batch size')
    parser.add_argument('--timesteps', type=int, default=2000, help='Rollout length')
    parser.add_argument('--model_path', type=str, default='actor_model.tflite')
    args = parser.parse_args()

    # 1) Load TFLite model and resize input for batch inference
    interpreter = tf.lite.Interpreter(model_path=args.model_path)
    inp_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]
    interpreter.resize_tensor_input(inp_det['index'], [args.num_envs, inp_det['shape'][-1]])
    interpreter.allocate_tensors()
    input_dim = inp_det['shape'][-1]

    # 2) Create the environment
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
    agent_names = env.agents

    # Try to get dt from environment or default
    try:
        dt = float(env.env.dt)
    except Exception:
        dt = 1.0 / 500.0

    # 3) Prepare RNG and reset batch
    rng = jax.random.PRNGKey(0)
    subkeys = jax.random.split(rng, args.num_envs + 1)
    rng = subkeys[0]
    reset_keys = subkeys[1:]
    obs_batch_dict, states = jax.vmap(env.reset)(reset_keys)
    obs = np.array(obs_batch_dict['global'], dtype=np.float32)[:, :input_dim]

    # 4) Initialize history buffers
    obs_history = []
    states_history = []
    actions_history = []

    # 5) Run rollouts
    for _ in range(args.timesteps):
        obs_history.append(obs.copy())
        # convert JAX states pytree to NumPy per step
        states_history.append(tree_map(lambda x: np.array(x), states))

        # model inference
        interpreter.set_tensor(inp_det['index'], obs)
        interpreter.invoke()
        action = interpreter.get_tensor(out_det['index'])
        actions_history.append(action.copy())

        # split or duplicate flat action into per-agent dict
        action_jax = jnp.array(action)
        total_dim = action_jax.shape[-1]
        global_act_size = env.env.action_size
        if total_dim * len(agent_names) == global_act_size:
            actions = {agent: action_jax for agent in agent_names}
        else:
            per_dim = total_dim // len(agent_names)
            actions = {
                agent_names[i]: action_jax[:, i * per_dim:(i+1) * per_dim]
                for i in range(len(agent_names))
            }

        # step environment with new PRNG keys
        subkeys = jax.random.split(rng, args.num_envs + 1)
        rng = subkeys[0]
        step_keys = subkeys[1:]
        obs_batch_dict, states, *_ = jax.vmap(
            env.step_env,
            in_axes=(0, 0, {agent: 0 for agent in agent_names})
        )(step_keys, states, actions)
        obs = np.array(obs_batch_dict['global'], dtype=np.float32)[:, :input_dim]

    # 6) Stack histories
    obs_history = np.stack(obs_history, axis=0)
    actions_history = np.stack(actions_history, axis=0)
    # use imported tree_map to stack state pytree history
    states_history = tree_map(lambda *arrs: np.stack(arrs, axis=0), *states_history)

    # compute final position errors
    pos_errs = np.linalg.norm(obs[:, :3], axis=1)

    # 7) Save data to ASDF file with metadata
    tree = {
        'metadata': {
            'dt': dt,
            'num_envs': args.num_envs,
            'timesteps': args.timesteps,
            'agent_names': agent_names
        },
        'obs_history': obs_history,
        'states_history': states_history,
        'actions_history': actions_history,
        'final_pos_errors': pos_errs,
        'timestep': np.arange(args.timesteps),
        'env_index': np.arange(args.num_envs)
    }
    af = asdf.AsdfFile(tree)
    af.write_to('flights.asdf')
    print("Saved all flight data and metadata to flights.asdf")

    # 8) Plot histogram
    plt.figure()
    plt.hist(pos_errs, bins=50)
    plt.xlabel('Position Error')
    plt.ylabel('Frequency')
    plt.title('Position Error Histogram at Last Frame')
    plt.savefig('pos_err_histogram.png')
    print("Saved histogram to pos_err_histogram.png")

if __name__ == "__main__":
    main()