"""
visualize_initial_conditions.py

This script loads the 'multiquad' environment, generates a batch of 100 initial states,
renders each state as a frame in a video (each frame lasting 1 second), and saves the result as a video file.
"""

import jax
import jax.numpy as jp
import imageio
import mujoco  # Used to create an OpenGL context
from brax import envs
import jaxmarl
import wandb
import time

def render_video(states, env, width=1280, height=720):  
    # Create an OpenGL context for rendering
    ctx = mujoco.GLContext(width, height)
    ctx.make_current()
    print("Rendering video...")
    frames = []
    for state in states:
        # Render a single frame for the given state
        frame = env.render([state], camera="track", width=width, height=height)[0]
        frames.append(frame)
    video_filename = "initial_conditions.mp4"
    # Each frame lasts 1 second (fps=1)
    imageio.mimsave(video_filename, frames, fps=1)
    #wandb.log({"initial_conditions_video": wandb.Video(video_filename, format="mp4")})
    print(f"Video saved to {video_filename}")

def main():
    # Minimal configuration for generating initial states
    config = {
        "ENV_NAME": "multiquad_2x4",
        "ENV_KWARGS": {},
        "SEED": 0,
    }

    rng = jax.random.PRNGKey(config["SEED"])
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    
    states = []
    # Generate a batch of 100 initial states
    for _ in range(100):
        rng, key = jax.random.split(rng)
        state = env.reset(key)[1]
        states.append(state)
    
    # Render video showing each state for 1 second
    render_video(states, env)

if __name__ == "__main__":
    main()