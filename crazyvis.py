#!/usr/bin/env python3
import os
import argparse
import numpy as np
import tensorflow as tf
import jax
import jax.numpy as jnp
import jaxmarl
import asdf

#------------------------------------------------------------------------------
# Vendored batchify / unbatchify (no external deps)
def batchify(x: dict, agent_list, num_actors):
    max_dim = max(x[a].shape[-1] for a in agent_list)
    def pad(z):
        return jnp.concatenate(
            [z, jnp.zeros(z.shape[:-1] + (max_dim - z.shape[-1],))],
            axis=-1
        )
    stacked = jnp.stack([
        x[a] if x[a].shape[-1] == max_dim else pad(x[a])
        for a in agent_list
    ])
    return stacked.reshape((num_actors, -1))

def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    split = x.reshape((len(agent_list), num_envs, -1))
    return {agent_list[i]: split[i] for i in range(len(agent_list))}

#------------------------------------------------------------------------------

# Setup XLA/MuJoCo cache
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"
cache_dir = os.path.join(os.getcwd(), "xla_cache")
os.makedirs(cache_dir, exist_ok=True)
os.environ["XLA_CACHE_DIR"] = cache_dir
jax.config.update("jax_compilation_cache_dir", cache_dir)

def parse_args():
    p = argparse.ArgumentParser(
        description="Run a multiquad batched rollout with a TFLite actor and dump to ASDF"
    )
    p.add_argument("--model_path", type=str, default="actor_model.tflite",
                   help="Path to the TFLite actor model")
    p.add_argument("--num_envs", type=int, default=100,
                   help="Number of parallel environments to vectorize")
    p.add_argument("--timesteps",  type=int, default=2500,
                   help="Number of simulation steps")
    p.add_argument("--output",     type=str, default="flights.crazy.asdf",
                   help="Filename for the ASDF output")
    return p.parse_args()

def load_model(model_path: str) -> tf.lite.Interpreter:
    return tf.lite.Interpreter(model_path=model_path)

def run_rollout(interpreter: tf.lite.Interpreter, num_envs: int, timesteps: int):
    # Build a single env and then vmap over it
    env = jaxmarl.make("multiquad_ix4")
    agents = env.agents
    obs_dim = env.observation_spaces[agents[0]].shape[0]

    # Prepare TFLite for batched inference
    interpreter.resize_tensor_input(
        interpreter.get_input_details()[0]["index"], [num_envs, obs_dim]
    )
    interpreter.allocate_tensors()
    inp_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]

    # Vectorized reset
    rng = jax.random.PRNGKey(0)
    rng, *reset_keys = jax.random.split(rng, num_envs + 1)
    reset_keys = jnp.stack(reset_keys)  # shape (num_envs,)

    vm_reset = jax.vmap(env.reset)
    obs_batched, pipeline_states = vm_reset(reset_keys)
    # obs_batched: dict of (num_envs, obs_dim); pipeline_states: tuple/list of length num_envs

    # Buffers
    obs_buf = []
    act_buf = []

    # Batched step function
    def batched_step(key, state, acts):
        return env.step_env(key, state, acts)
    # Now vmap over ALL three arguments
    vm_step = jax.vmap(batched_step, in_axes=(0, 0, 0))

    for _ in range(timesteps):
        # (1) record observations
        obs_arr = np.stack([np.array(obs_batched[a]) for a in agents], axis=1)
        # shape = (num_envs, num_agents, obs_dim)
        obs_buf.append(obs_arr)

        # (2) TFLite inference
        raw_acts = {}
        for a in agents:
            inp = np.array(obs_batched[a], dtype=np.float32)  # (num_envs, obs_dim)
            interpreter.set_tensor(inp_det["index"], inp)
            interpreter.invoke()
            raw_acts[a] = jnp.array(interpreter.get_tensor(out_det["index"]))  # (num_envs, act_dim)
        act_arr = np.stack([np.array(raw_acts[a]) for a in agents], axis=1)
        # shape = (num_envs, num_agents, act_dim)
        act_buf.append(act_arr)

        # (3) step in parallel
        rng, *step_keys = jax.random.split(rng, num_envs + 1)
        step_keys = jnp.stack(step_keys)
        obs_batched, pipeline_states, _, _, _ = vm_step(step_keys, pipeline_states, raw_acts)

    # Stack time dimension: (timesteps, num_envs, num_agents, dim)
    obs_history = np.stack(obs_buf, axis=0)
    act_history = np.stack(act_buf, axis=0)
    return obs_history, act_history, agents

def save_rollout(obs_h, act_h, agents, args, num_envs):
    """
    Save batched rollout to ASDF with:
      agents: name -> {
        observations: ndarray[timesteps, num_envs, obs_dim],
        actions:      ndarray[timesteps, num_envs, act_dim]
      }
    """
    agents_dict = {
        agents[i]: {
            "observations": obs_h[:, :, i, :],
            "actions":      act_h[:, :, i, :]
        }
        for i in range(len(agents))
    }

    tree = {
        "metadata": {
            "num_envs":    num_envs,
            "timesteps":   args.timesteps,
            "agent_names": agents
        },
        "environment": {"name": "multiquad_ix4"},
        "flights": [{
            "agents": agents_dict
        }]
    }

    af = asdf.AsdfFile(tree)
    af.write_to(args.output)
    print(f"Saved ASDF to {args.output}")

def main():
    args = parse_args()
    interpreter = load_model(args.model_path)
    obs_h, act_h, agents = run_rollout(interpreter, args.num_envs, args.timesteps)
    save_rollout(obs_h, act_h, agents, args, args.num_envs)

if __name__ == "__main__":
    main()