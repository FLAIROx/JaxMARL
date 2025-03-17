"""
run_multiquad_random_video.py

This script loads the 'multiquad' environment, runs 2500 simulation steps with random actions,
renders the rollout (rendering every few frames), and saves the result as a video file.
"""

import jax
import jax.numpy as jp
import imageio
import mujoco  # Used to create an OpenGL context
from brax import envs
import jaxmarl
import time
import wandb
# Import training utilities and network definitions from ippo_ff_mabrax.py
from baselines.IPPO.ippo_ff_mabrax import make_train, ActorCritic, batchify, unbatchify

# New function to render and save video
def render_video(rollout, env, render_every=2, width=1280, height=720):
    # Create an OpenGL context for rendering
    ctx = mujoco.GLContext(width, height)
    ctx.make_current()
    print("Starting rendering...")
    frames = env.render(rollout[::render_every], camera="track", width=width, height=height)
    fps = float(1.0 / (env.dt * render_every))
    # Changed video filename as per previous code
    video_filename = "trained_policy_video.mp4"
    imageio.mimsave(video_filename, frames, fps=fps)
    # New wandb logging for the video
    wandb.log({"trained_policy_video": wandb.Video(video_filename, format="mp4")})
    print(f"Video saved to {video_filename}")

def main():
    # Build configuration for IPPO training on multiquad_2x4
    config = {
        "ENV_NAME": "multiquad_2x4",
        "ENV_KWARGS": {},
        "TOTAL_TIMESTEPS": 10_000_000,
        "NUM_ENVS": 1024,
        "NUM_STEPS": 2048,
        "NUM_MINIBATCHES": 8,
        "UPDATE_EPOCHS": 2,
        "ANNEAL_LR": False,
        "LR": 3e-4,
        "ACTIVATION": "tanh",
        "MAX_GRAD_NORM": 0.5,
        "CLIP_EPS": 0.2,
        "VF_COEF": 0.5,
        "ENT_COEF": 0.01,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "SEED": 0,
        "DISABLE_JIT": False,
        "PROJECT": "single_quad_rl",
        "NAME": f"quad_marl_{int(time.time())}",
        "WANDB_MODE": "online"
    }
    wandb.init(
        name=config["NAME"],
        project=config["PROJECT"],
        tags=["IPPO", "FF"],
        config=config,
        mode=config["WANDB_MODE"],
    )
    
    rng = jax.random.PRNGKey(config["SEED"])
    rng, rng_train = jax.random.split(rng)
    
    # Train the policy using IPPO training routine from ippo_ff_mabrax.py
    train_fn = make_train(config, rng_train)
    out = train_fn(rng)
    # Extract the trained train_state (the first element of runner_state)
    train_state = out["runner_state"][0]
    
    # Create the multiquad environment without the jaxmarl wrapper
    from brax import envs
    env = envs.get_environment("multiquad")
    
    # Initialize network and policy function for prediction.
    obs_shape = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    network = ActorCritic(action_dim=act_dim, activation=config["ACTIVATION"])

    def policy_fn(params, obs, key):
        batched_obs = batchify(obs, env.agents, env.num_agents)
        pi, _ = network.apply(params, batched_obs)
        actions = pi.sample(seed=key)
        unbatched = unbatchify(actions, env.agents, 1, env.num_agents)
        # Squeeze the batch dimension for each agent's action.
        unbatched = {a: jp.squeeze(act, axis=0) for a, act in unbatched.items()}
        return unbatched

    # Simulation loop using policy prediction on the native multiquad env
    rng, rng_sim = jax.random.split(rng)
    state = env.reset(rng_sim)
    rollout = [state.pipeline_state]
    jit_step = jax.jit(env.step)
    n_steps = 2500
    print("Starting simulation with trained policy...")
    for i in range(n_steps):
        rng, key = jax.random.split(rng)
        # Use state's obs directly
        actions = policy_fn(train_state.params, state.obs, key)
        state = jit_step(state, actions)
        rollout.append(state.pipeline_state)
    state = jax.block_until_ready(state)
    print("Simulation finished.")


    
    # Rendering code (creates OpenGL context, renders frames, and saves video)
    ctx = mujoco.GLContext(1280, 720)
    ctx.make_current()
    render_every = 2
    print("Starting rendering...")
    frames = env.render(rollout[::render_every], camera="track", width=1280, height=720)
    fps = float(1.0 / (env.dt * render_every))
    video_filename = "rollout_video.mp4"
    imageio.mimsave(video_filename, frames, fps=fps)
    print(f"Video saved to {video_filename}")
    
if __name__ == "__main__":
    main()