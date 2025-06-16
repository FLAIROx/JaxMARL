#!/usr/bin/env python3
import os
import argparse
import numpy as np
import tensorflow as tf
import jax
import jax.numpy as jnp
import jaxmarl
import asdf
from jax import tree_util, Array  # for pytree operations and Array type

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

# Setup XLA/MuJoCo cache
env_cache = os.path.join(os.getcwd(), "xla_cache")
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"
os.makedirs(env_cache, exist_ok=True)
os.environ["XLA_CACHE_DIR"] = env_cache
jax.config.update("jax_compilation_cache_dir", env_cache)

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
    env = jaxmarl.make("multiquad_ix4")
    agents = env.agents
    obs_dim = env.observation_spaces[agents[0]].shape[0]

    interpreter.resize_tensor_input(
        interpreter.get_input_details()[0]["index"], [num_envs, obs_dim]
    )
    interpreter.allocate_tensors()
    inp_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]

    rng = jax.random.PRNGKey(0)
    rng, *reset_keys = jax.random.split(rng, num_envs + 1)
    reset_keys = jnp.stack(reset_keys)

    vm_reset = jax.vmap(env.reset)
    obs_batched, pipeline_states = vm_reset(reset_keys)

    obs_buf, act_buf, rew_buf, done_buf = [], [], [], []
    vm_step = jax.vmap(lambda k, s, a: env.step_env(k, s, a), in_axes=(0, 0, 0))

    for _ in range(timesteps):
        obs_arr = np.stack([np.array(obs_batched[a]) for a in agents], axis=1)
        obs_buf.append(obs_arr)

        raw_acts = {}
        for a in agents:
            inp = np.array(obs_batched[a], dtype=np.float32)
            interpreter.set_tensor(inp_det["index"], inp)
            interpreter.invoke()
            raw_acts[a] = jnp.array(interpreter.get_tensor(out_det["index"]))
        act_arr = np.stack([np.array(raw_acts[a]) for a in agents], axis=1)
        act_buf.append(act_arr)

        rng, *step_keys = jax.random.split(rng, num_envs + 1)
        step_keys = jnp.stack(step_keys)
        obs_batched, pipeline_states, rewards, dones, _ = vm_step(
            step_keys, pipeline_states, raw_acts
        )

        rew_arr = np.stack([np.array(rewards[a]) for a in agents], axis=1)
        rew_buf.append(np.array(rewards["__all__"]))
        done_buf.append(np.array(dones["__all__"]))

    obs_history = np.stack(obs_buf, axis=0)
    act_history = np.stack(act_buf, axis=0)
    rew_history = np.stack(rew_buf, axis=0)
    done_history = np.stack(done_buf, axis=0)

    return obs_history, act_history, rew_history, done_history, pipeline_states, agents

def save_rollout(obs_h, act_h, rew_h, done_h, pipeline_states, agents, args, num_envs):
    # Build per-agent data using numpy for slicing
    agents_dict = {
        agents[i]: {
            "observations": obs_h[:, :, i, :],
            "actions":      act_h[:, :, i, :],
        }
        for i in range(len(agents))
    }

    tree = {
        "metadata": {"num_envs": num_envs, "timesteps": args.timesteps, "agent_names": agents},
        "environment": {"name": "multiquad_ix4"},
        "flights": [{
            "metadata": {"num_envs": num_envs, "timesteps": args.timesteps,
                          "agents": agents, "model_path": args.model_path},
            "agents": agents_dict,
            "global": {
                        "rewards": rew_h, 
                        "dones": done_h,  # Global done status
                      # "pipeline_states": pipeline_states
                       }
        }]
    }

    # # Convert all numpy and JAX arrays to Python lists for ASDF serialization
    # def _to_list(x):
    #     if isinstance(x, (np.ndarray, Array)):
    #         return np.array(x).tolist()
    #     return x
    # tree = tree_util.tree_map(_to_list, tree)

    af = asdf.AsdfFile(tree)
    af.write_to(args.output)
    print(f"Saved ASDF to {args.output}")

    

def main():
    args = parse_args()
    interpreter = load_model(args.model_path)

    obs_h, act_h, rew_h, done_h, pipeline_states, agents = \
        run_rollout(interpreter, args.num_envs, args.timesteps)

    save_rollout(obs_h, act_h, rew_h, done_h,
                 pipeline_states, agents, args, args.num_envs)

if __name__ == "__main__":
    main()