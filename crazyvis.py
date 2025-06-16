#!/usr/bin/env python3
import os
import argparse
import numpy as np
import tensorflow as tf
import jax
import jax.numpy as jnp
import jaxmarl
import asdf
from jax.tree_util import tree_map

#------------------------------------------------------------------------------
# Vendored batchify / unbatchify
def batchify(x: dict, agent_list, num_actors):
    max_dim = max(x[a].shape[-1] for a in agent_list)
    def pad(z):
        return jnp.concatenate([z, jnp.zeros(z.shape[:-1] + (max_dim - z.shape[-1],))], -1)
    stacked = jnp.stack([
        x[a] if x[a].shape[-1] == max_dim else pad(x[a])
        for a in agent_list
    ])
    return stacked.reshape((num_actors, -1))

def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    split = x.reshape((len(agent_list), num_envs, -1))
    return {agent_list[i]: split[i] for i in range(len(agent_list))}
#------------------------------------------------------------------------------

# XLA / MuJoCo cache setup
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"
cache_dir = os.path.join(os.getcwd(), "xla_cache")
os.makedirs(cache_dir, exist_ok=True)
os.environ["XLA_CACHE_DIR"] = cache_dir
jax.config.update("jax_compilation_cache_dir", cache_dir)

def parse_args():
    p = argparse.ArgumentParser(
        description="Run a multiquad rollout with a TFLite actor and dump to ASDF"
    )
    p.add_argument("--model_path", type=str, default="actor_model.tflite",
                   help="Path to the TFLite actor model")
    p.add_argument("--timesteps",  type=int, default=2500,
                   help="Number of simulation steps")
    p.add_argument("--output",     type=str, default="flights.crazy.asdf",
                   help="Filename for the ASDF output")
    return p.parse_args()

def load_model(model_path: str) -> tf.lite.Interpreter:
    interpreter = tf.lite.Interpreter(model_path=model_path)
    return interpreter

def run_rollout(interpreter: tf.lite.Interpreter, timesteps: int):
    # 1) single‐env creation
    env    = jaxmarl.make("multiquad_ix4")      # no num_envs here  [oai_citation:3‡kngwyu.github.io](https://kngwyu.github.io/rlog/ja/2021/12/18/jax-brax-haiku.html?utm_source=chatgpt.com)
    agents = env.agents
    obs_dim = env.observation_spaces[agents[0]].shape[0]

    # 2) prepare TFLite for batch=1
    interpreter.resize_tensor_input(
        interpreter.get_input_details()[0]["index"],
        [1, obs_dim]
    )
    interpreter.allocate_tensors()
    inp_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]

    obs_buf, act_buf, state_buf = [], [], []

    # 3) RESET → unpack tuple
    rng = jax.random.PRNGKey(0)
    rng, reset_key = jax.random.split(rng)
    obs_dict, pipeline_state = env.reset(reset_key)  # returns (obs, pipeline_state)

    for t in range(timesteps):
        # a) RECORD obs
        obs_buf.append( np.stack([ np.array(obs_dict[a]) for a in agents ], axis=0) )

        # b) ACT: one‐agent TFLite inference per agent
        raw_acts = {}
        for a in agents:
            inp = np.array(obs_dict[a][None, :], dtype=np.float32)
            interpreter.set_tensor(inp_det["index"], inp)
            interpreter.invoke()
            out = interpreter.get_tensor(out_det["index"])[0]
            raw_acts[a] = jnp.array(out)
        act_buf.append( np.stack([ np.array(raw_acts[a]) for a in agents ], axis=0) )

        # c) RECORD MJX state by iterating over all array fields
        flat = {}
        for name in dir(pipeline_state):
            if name.startswith("_"): continue
            val = getattr(pipeline_state, name)
            if isinstance(val, (np.ndarray, jnp.ndarray)):
                flat[name] = np.array(val)
        state_buf.append(flat)

        # d) STEP → unpack tuple
        obs_dict, pipeline_state, reward, dones, info = env.step_env(
            jax.random.split(rng)[1], pipeline_state, raw_acts
        )

    # stack: (T, num_agents, dim)
    obs_history     = np.stack(obs_buf, axis=0)
    actions_history = np.stack(act_buf, axis=0)
    return obs_history, actions_history, state_buf, agents

def save_rollout(obs_h, act_h, state_buf, agents, args):
    # build states dict: each field shape (T, *field_shape)
    fields = list(state_buf[0].keys())
    states_dict = {
        f: np.stack([ step[f] for step in state_buf ], axis=0)
        for f in fields
    }

    tree = {
        "metadata": {
            "num_envs":    1,
            "timesteps":   args.timesteps,
            "agent_names": agents
        },
        "environment": {"name": "multiquad_ix4"},
        "flights": [{
            "states": states_dict,
            "agents": {
                agents[i]: {
                    "observations": obs_h[:, i],
                    "actions":      act_h[:, i]
                } for i in range(len(agents))
        }
        }]
    }
    af = asdf.AsdfFile(tree)
    af.write_to(args.output)
    print(f"Saved ASDF to {args.output}")

def main():
    args = parse_args()
    interpreter = load_model(args.model_path)
    obs_h, act_h, state_buf, agents = run_rollout(interpreter, args.timesteps)
    save_rollout(obs_h, act_h, state_buf, agents, args)

if __name__ == "__main__":
    main()