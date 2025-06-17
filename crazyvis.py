#!/usr/bin/env python3
import os
import argparse
import numpy as np
import tensorflow as tf
import jax
import jax.numpy as jnp
import jaxmarl
import asdf
import imageio
import mujoco  # for OpenGL context

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
# Rendering utility
def render_video(pipeline_states, env, render_every=10, width=640, height=480, output="rollout_video.mp4"):
    ctx = mujoco.GLContext(width, height)
    ctx.make_current()
    print("Rendering rollout...")
    frames = env.render(pipeline_states[::render_every], camera="track", width=width, height=height)
    fps = float(1.0 / (env.dt * render_every))
    imageio.mimsave(output, frames, fps=fps)
    print(f"Video saved to {output}")

#------------------------------------------------------------------------------
# Setup XLA/MuJoCo cache
env_cache = os.path.join(os.getcwd(), "xla_cache")
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"
os.makedirs(env_cache, exist_ok=True)
os.environ["XLA_CACHE_DIR"] = env_cache
jax.config.update("jax_compilation_cache_dir", env_cache)

#------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Run a multiquad rollout with a TFLite actor, save to ASDF, and render video"
    )
    p.add_argument("--model_path", type=str, default="actor_model.tflite", help="Path to TFLite actor model")
    p.add_argument("--num_envs", type=int, default=1000, help="Number of parallel environments")
    p.add_argument("--timesteps", type=int, default=4000, help="Number of simulation steps")
    p.add_argument("--output", type=str, default="flights.crazy.asdf", help="ASDF output filename")
    p.add_argument("--video", type=str, default="rollout_video.mp4", help="Rendered video filename")
    return p.parse_args()

#------------------------------------------------------------------------------
def load_model(model_path: str) -> tf.lite.Interpreter:
    interpreter = tf.lite.Interpreter(model_path=model_path)
    return interpreter

#------------------------------------------------------------------------------
def run_batched_rollout(interpreter: tf.lite.Interpreter, num_envs: int, timesteps: int):
    env = jaxmarl.make("multiquad_ix4")
    agents = env.agents
    obs_dim = env.observation_spaces[agents[0]].shape[0]

    interpreter.resize_tensor_input(
        interpreter.get_input_details()[0]["index"], [num_envs, obs_dim]
    )
    interpreter.allocate_tensors()
    inp_idx = interpreter.get_input_details()[0]["index"]
    out_idx = interpreter.get_output_details()[0]["index"]

    # Batch reset to get initial states
    rng = jax.random.PRNGKey(0)
    keys = jax.random.split(rng, num_envs)
    obs_batched, pipeline_states = jax.vmap(env.reset)(keys)

    obs_buf, act_buf, rew_buf, done_buf = [], [], [], []
    vm_step = jax.vmap(env.step_env, in_axes=(0, 0, 0))

    for _ in range(timesteps):
        # Record observations
        obs_stack = np.stack([np.array(obs_batched[a]) for a in agents], axis=1)
        obs_buf.append(obs_stack)

        # Actor inference
        raw_acts = {}
        for a in agents:
            inp = np.array(obs_batched[a], dtype=np.float32)
            interpreter.set_tensor(inp_idx, inp)
            interpreter.invoke()
            raw_acts[a] = jnp.array(interpreter.get_tensor(out_idx))
        act_stack = np.stack([np.array(raw_acts[a]) for a in agents], axis=1)
        act_buf.append(act_stack)

        # Step environments
        rng, *step_keys = jax.random.split(rng, num_envs + 1)
        step_keys = jnp.stack(step_keys)
        obs_batched, pipeline_states, rewards, dones, _ = vm_step(
            step_keys, pipeline_states, raw_acts
        )
        rew_buf.append(np.array(rewards["__all__"]))
        done_buf.append(np.array(dones["__all__"]))

    return (
        np.stack(obs_buf),
        np.stack(act_buf),
        np.stack(rew_buf),
        np.stack(done_buf)
    )

#------------------------------------------------------------------------------
def run_single_rollout(interpreter: tf.lite.Interpreter, timesteps: int):
    env = jaxmarl.make("multiquad_ix4")
    agents = env.agents
    obs_dim = env.observation_spaces[agents[0]].shape[0]

    interpreter.resize_tensor_input(
        interpreter.get_input_details()[0]["index"], [1, obs_dim]
    )
    interpreter.allocate_tensors()
    inp_idx = interpreter.get_input_details()[0]["index"]
    out_idx = interpreter.get_output_details()[0]["index"]

    rng = jax.random.PRNGKey(1)
    obs, state = env.reset(rng)
    states = []
    for _ in range(timesteps):
        states.append(state)
        # Per-agent inference
        actions = {}
        for a in agents:
            ob = np.expand_dims(obs[a], axis=0).astype(np.float32)
            interpreter.set_tensor(inp_idx, ob)
            interpreter.invoke()
            out = interpreter.get_tensor(out_idx)
            actions[a] = jnp.squeeze(out, axis=0)
        # Step single environment
        rng, key = jax.random.split(rng)
        obs, state, _, dones, _ = env.step_env(key, state, actions)
        # if any done
        if any(dones.values()):
            print("Resetting environment due to done state")
            rng, key = jax.random.split(rng)
            obs, state = env.reset(key)
    return states

#------------------------------------------------------------------------------
def save_rollout(obs_h, act_h, rew_h, done_h, args, num_envs, agents):
    agents_dict = {
        agents[i]: {"observations": obs_h[:, :, i, :], "actions": act_h[:, :, i, :]}
        for i in range(len(agents))
    }
    tree = {
        "metadata": {"num_envs": num_envs, "timesteps": obs_h.shape[0], "agent_names": agents},
        "environment": {"name": "multiquad_ix4"},
        "flights": [{
            "metadata": {"num_envs": num_envs, 
                         "timesteps": obs_h.shape[0], 
                         "agents": agents,
                        "model_path": args.model_path,
                           "dt": 1.0/250,
                           "env": "multiquad_ix4"
                           },
            "agents": agents_dict,
            "global": {"rewards": rew_h, "dones": done_h}
        }]
    }
    asdf.AsdfFile(tree).write_to(args.output)
    print(f"Saved ASDF to {args.output}")

#------------------------------------------------------------------------------
def main():
    args = parse_args()
    interpreter = load_model(args.model_path)

    # Batched rollout: collect and save data
    obs_h, act_h, rew_h, done_h = run_batched_rollout(interpreter, args.num_envs, args.timesteps)
    env = jaxmarl.make("multiquad_ix4")
    agents = env.agents
    save_rollout(obs_h, act_h, rew_h, done_h, args, args.num_envs, agents)

    # Single-env rollout for rendering
    print("Running single-env rollout for rendering...")
    states = run_single_rollout(interpreter, args.timesteps)
    render_video(states, env, render_every=10, width=640, height=480, output=args.video)

if __name__ == "__main__":
    main()
