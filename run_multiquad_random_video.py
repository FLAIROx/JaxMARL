"""
run_multiquad_random_video.py

This script loads the 'multiquad' environment, runs 2500 simulation steps with random actions,
renders the rollout (rendering every few frames), and saves the result as a video file.
"""

import jax
import jax.numpy as jp
from brax import envs
import imageio
import mujoco  # Used to create an OpenGL context
import multi_quad_env  # Ensure your custom environment file is in your PYTHONPATH

def main():
    # Load the environment by its registered name.
    env = envs.get_environment('multiquad')
    
    # Initialize the environment.
    rng = jax.random.PRNGKey(0)
    state = env.reset(rng)
    
    # Simulation parameters.
    n_steps = 2500
    render_every = 2  # Render every 2nd frame (to reduce video length)
    rollout = [state.pipeline_state]
    
    # Run simulation for n_steps with random actions.
    for i in range(n_steps):
        rng, key = jax.random.split(rng)
        # Sample random action in range [-1, 1] with the appropriate shape.
        action = jax.random.uniform(key, shape=(env.sys.nu,), minval=-1.0, maxval=1.0)
        state = env.step(state, action)
        rollout.append(state.pipeline_state)
    
    # Create an OpenGL context (using mujoco.GLContext) with desired resolution.
    ctx = mujoco.GLContext(1920, 1080)
    ctx.make_current()
    
    # Render only every render_every-th frame.
    frames = env.render(rollout[::render_every], camera="track", width=1920, height=1080)
    
    # Calculate frames per second: original code used fps=1.0 / (env.dt * render_every).
    fps = 1.0 / (env.dt * render_every)
    video_filename = "rollout_video.mp4"
    
    # Save frames as video using imageio.
    imageio.mimsave(video_filename, frames, fps=fps)
    print(f"Video saved to {video_filename}")

if __name__ == "__main__":
    main()