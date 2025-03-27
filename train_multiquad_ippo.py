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
        "TOTAL_TIMESTEPS": 200_000_000,
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
    train_fn = jax.jit(make_train(config, rng_train))
    out = train_fn(rng)
    # Extract the trained train_state (the first element of runner_state)
    train_state = out["runner_state"][0]
    
    # Create the environment for simulation
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    
    # Initialize the ActorCritic network with proper dimensions (use first agent's spaces)
    obs_shape = env.observation_spaces[env.agents[0]].shape[0]
    act_dim = env.action_spaces[env.agents[0]].shape[0]
    network = ActorCritic(action_dim=act_dim, activation=config["ACTIVATION"])
    
    # Define a policy function to map observations to actions using the trained parameters
    def policy_fn(params, obs, key):
        batched_obs = batchify(obs, env.agents, env.num_agents)
        pi, _ = network.apply(params, batched_obs)
        actions = pi.sample(seed=key)
        unbatched = unbatchify(actions, env.agents, 1, env.num_agents)
        # Squeeze the extra batch dimension so each agent's action has shape (action_dim,)
        unbatched = {a: jp.squeeze(act, axis=0) for a, act in unbatched.items()}
        return unbatched
    
    # Simulation: run an episode using the trained policy
    sim_steps = 4000
    rng, rng_sim = jax.random.split(rng)
    state = env.reset(rng_sim)
    rollout = [state[1]]  # store the brax env state
    
    print("Starting simulation with trained policy...")
    for i in range(sim_steps):
        rng, key = jax.random.split(rng)
        actions = policy_fn(train_state.params, env.get_obs(state[1]), key)  
        rng, key = jax.random.split(rng)
        # Unpack but use new_state from env.step_env
        _, new_state, rewards, dones, info = env.step_env(key, state[1], actions)
        rollout.append(new_state)
        state = (None, new_state)
    state = jax.block_until_ready(state)
    print("Simulation finished.")
    
    # Call the separated video rendering function
    render_video(rollout, env)
    
if __name__ == "__main__":
    main()