"""
run_multiquad_random_video.py

This script loads the 'multiquad' environment, runs 2500 simulation steps with random actions,
renders the rollout (rendering every few frames), and saves the result as a video file.
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"
# Create a cache directory relative to the current working directory
cache_dir = os.path.join(os.getcwd(), "xla_cache")
os.makedirs(cache_dir, exist_ok=True)

# Set the XLA cache directory to this folder
os.environ["XLA_CACHE_DIR"] = cache_dir

# Nvlink version mismatch fix
# This is a workaround for a known issue with JAX and NVLink on some systems.
# os.environ["TF_USE_NVLINK_FOR_PARALLEL_COMPILATION"] = "0"
# os.environ["XLA_FLAGS"] = "--xla_gpu_force_compilation_parallelism=1"

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

from jax2onnx import save_onnx

# Set JAX cache
jax.config.update("jax_compilation_cache_dir", cache_dir)
# jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
# jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
# jax.config.update("jax_persistent_cache_enable_xla_caches", "all")


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


def eval_results(eval_env, jit_reset, jit_inference_fn, jit_step):
    import numpy as np
    import matplotlib.pyplot as plt
    import io
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from PIL import Image
    import matplotlib.ticker as mticker
    from matplotlib.colors import LinearSegmentedColormap

    # --------------------
    # Simulation
    # --------------------
    n_steps = 2500
    render_every = 2
    rng = jax.random.PRNGKey(0)
    state = jit_reset(rng)
    rollout = [state.pipeline_state]
    quad_actions_list = []
    for i in range(n_steps):
        act_rng, rng = jax.random.split(rng)
        ctrl, _ = jit_inference_fn(state.obs, act_rng)
        state = jit_step(state, ctrl)
        rollout.append(state.pipeline_state)
        quad_actions_list.append(np.array(ctrl))
    # Skipping video rendering since it is handled separately.
    
    # Histogram plot over quad actions.
    quad_actions_flat = np.concatenate(quad_actions_list).flatten()
    plt.figure()
    plt.hist(quad_actions_flat, bins=50)
    plt.xlabel('Action Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Quad Actions')
    plt.savefig('quad_actions_histogram.png')
    print("Plot saved: quad_actions_histogram.png")
    wandb.log({"quad_actions_histogram": wandb.Image('quad_actions_histogram.png')})
    plt.close()

    # --------------------
    # 3D Trajectory Plot for Payload
    # --------------------
    payload_id = eval_env.payload_body_id
    payload_positions = [np.array(s.xpos[payload_id]) for s in rollout]
    payload_positions = np.stack(payload_positions)
    
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(payload_positions[:,0], payload_positions[:,1], payload_positions[:,2],
            label='Payload Trajectory', lw=2)
    goal = np.array(eval_env.target_position)
    ax.scatter(goal[0], goal[1], goal[2], color='red', s=50, label='Goal Position')
    start_pos = payload_positions[0]
    ax.scatter(start_pos[0], start_pos[1], start_pos[2], color='green', s=50, label='Start Position')
    
    quad1_positions = np.stack([np.array(s.xpos[eval_env.q1_body_id]) for s in rollout])
    quad2_positions = np.stack([np.array(s.xpos[eval_env.q2_body_id]) for s in rollout])
    
    ax.plot(quad1_positions[:,0], quad1_positions[:,1], quad1_positions[:,2],
            ls='--', color='blue', lw=2, alpha=0.5, label='Quad1 Trajectory')
    ax.plot(quad2_positions[:,0], quad2_positions[:,1], quad2_positions[:,2],
            ls='--', color='magenta', lw=2, alpha=0.5, label='Quad2 Trajectory')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title('Payload Trajectory')
    ax.xaxis.set_major_locator(mticker.MaxNLocator(5))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(5))
    ax.zaxis.set_major_locator(mticker.MaxNLocator(5))
    ax.set_zlim(0, 1.5)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300)
    print("Plot saved: 3D payload trajectory plot")
    buf.seek(0)
    img = Image.open(buf)
    wandb.log({"payload_trajectory": wandb.Image(img)})
    plt.close(fig)
    
    # --------------------
    # Top-Down (XY) Trajectory Plot for Payload
    # --------------------
    fig_topdown = plt.figure(figsize=(5, 5))
    plt.plot(payload_positions[:,0], payload_positions[:,1],
             label='Payload XY Trajectory', lw=2)
    plt.plot(quad1_positions[:,0], quad1_positions[:,1],
             ls='--', color='blue', lw=2, alpha=0.7, label='Quad1 XY Trajectory')
    plt.plot(quad2_positions[:,0], quad2_positions[:,1],
             ls='--', color='magenta', lw=2, alpha=0.7, label='Quad2 XY Trajectory')
    plt.scatter(goal[0], goal[1], color='red', s=50, label='Goal Position')
    plt.scatter(start_pos[0], start_pos[1], color='green', s=50, label='Start Position')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Payload Trajectory (Top Down)')
    plt.legend()
    buf_top = io.BytesIO()
    plt.savefig(buf_top, format='png', dpi=300)
    print("Plot saved: Top-down payload trajectory plot")
    buf_top.seek(0)
    img_top = Image.open(buf_top)
    wandb.log({"payload_trajectory_topdown": wandb.Image(img_top)})
    plt.close(fig_topdown)
    
    # --------------------
    # Payload Position Error Over Time Plot
    # --------------------
    times_sim = np.array([s.time for s in rollout])
    payload_errors = np.array([np.linalg.norm(np.array(s.xpos[payload_id]) - np.array(eval_env.target_position))
                               for s in rollout])
    fig2 = plt.figure()
    plt.plot(times_sim, payload_errors, linestyle='-', color='orange', label='Payload Error')
    plt.xlabel('Simulation Time (s)')
    plt.ylabel('Payload Position Error')
    plt.title('Payload Position Error Over Time')
    plt.legend()
    buf2 = io.BytesIO()
    plt.savefig(buf2, format='png', dpi=300)
    print("Plot saved: Payload error over time plot")
    buf2.seek(0)
    img2 = Image.open(buf2)
    wandb.log({"payload_error_over_time": wandb.Image(img2)})
    plt.close(fig2)
    
    # --------------------
    # Batched Rollout over 100 Envs and Top-Down XY Plot for Final Positions 
    # --------------------
    num_envs = 100
    n_steps = 2500
    batched_rngs = jax.random.split(jax.random.PRNGKey(1234), num_envs)
    batched_states = jax.vmap(jit_reset)(batched_rngs)
    
    start_positions = np.array(jax.vmap(lambda s: s.pipeline_state.xpos[eval_env.payload_body_id])(batched_states))
    
    batched_errors = []
    timeline = []
    rng_main = jax.random.PRNGKey(5678)
    for step in range(n_steps):
        rng_main, rng_step = jax.random.split(rng_main)
        act_rngs = jax.random.split(rng_step, num_envs)
        ctrls, _ = jax.vmap(jit_inference_fn)(batched_states.obs, act_rngs)
        batched_states = jax.vmap(jit_step)(batched_states, ctrls)
        errors = jax.vmap(lambda s: jax.numpy.linalg.norm(s.pipeline_state.xpos[eval_env.payload_body_id] - eval_env.target_position))(batched_states)
        batched_errors.append(np.array(errors))
        times_env = jax.vmap(lambda s: s.pipeline_state.time)(batched_states)
        timeline.append(np.array(times_env[0]))
    
    final_payload_positions = np.array(jax.vmap(lambda s: s.pipeline_state.xpos[eval_env.payload_body_id])(batched_states))[:, :2]
    final_quad1_positions = np.array(jax.vmap(lambda s: s.pipeline_state.xpos[eval_env.q1_body_id])(batched_states))[:, :2]
    final_quad2_positions = np.array(jax.vmap(lambda s: s.pipeline_state.xpos[eval_env.q2_body_id])(batched_states))[:, :2]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(start_positions[:, 0], start_positions[:, 1],
               color='black', s=10, label='Start Payload')
    ax.scatter(goal[0], goal[1], color='red', s=70, marker='*', label='Goal Position')
    new_cmap = LinearSegmentedColormap.from_list('custom_cmap', [(0, (1,1,1,0)), (1, (0,0,1,1))], N=256)
    x_low, x_high = -0.3, 0.3
    y_low, y_high = -0.3, 0.3
    inliers_mask = ((final_payload_positions[:, 0] >= x_low) &
                    (final_payload_positions[:, 0] <= x_high) &
                    (final_payload_positions[:, 1] >= y_low) &
                    (final_payload_positions[:, 1] <= y_high))
    inliers = final_payload_positions[inliers_mask]
    outliers = final_payload_positions[~inliers_mask]
    xbins = np.linspace(x_low, x_high, 30)
    ybins = np.linspace(y_low, y_high, 30)
    H, xedges, yedges = np.histogram2d(inliers[:, 0], inliers[:, 1], bins=[xbins, ybins], density=True)
    Xc = (xedges[:-1] + xedges[1:]) / 2
    Yc = (yedges[:-1] + yedges[1:]) / 2
    X, Y = np.meshgrid(Xc, Yc)
    cont = ax.contourf(X, Y, H.T, levels=10, cmap=new_cmap, alpha=0.7, vmin=0)
    cbar = fig.colorbar(cont, ax=ax)
    cbar.set_label('Density')
    if outliers.size > 0:
        ax.scatter(outliers[:, 0], outliers[:, 1], color='cyan', marker='x', s=20, label='Outliers')
    ax.set_xlim(x_low, x_high)
    ax.set_ylim(y_low, y_high)
    ax.scatter(final_quad1_positions[:, 0], final_quad1_positions[:, 1],
               color='blue', marker='s', s=15, alpha=0.3, label='Quad1 Final')
    ax.scatter(final_quad2_positions[:, 0], final_quad2_positions[:, 1],
               color='magenta', marker='s', s=15, alpha=0.3, label='Quad2 Final')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Top-Down XY Plot for Final Positions (Batched Rollout)')
    ax.legend()
    buf_final = io.BytesIO()
    plt.savefig(buf_final, format='png', dpi=300)
    buf_final.seek(0)
    img_final = Image.open(buf_final)
    wandb.log({"batched_rollout_topdown": wandb.Image(img_final)})
    print("Plot saved and logged: batched_rollout_topdown")
    plt.close(fig)
    
    # --------------------
    # Batched Payload Error Over Time Plot using percentiles
    # --------------------
    timeline = np.array(timeline)
    batched_errors = np.array(batched_errors)
    p0 = np.percentile(batched_errors, 0, axis=1)
    p25 = np.percentile(batched_errors, 25, axis=1)
    p50 = np.percentile(batched_errors, 50, axis=1)
    p75 = np.percentile(batched_errors, 75, axis=1)
    p90 = np.percentile(batched_errors, 90, axis=1)
    p98 = np.percentile(batched_errors, 98, axis=1)
    p100 = np.percentile(batched_errors, 100, axis=1)
    
    fig3 = plt.figure(figsize=(8, 5))
    ax3 = fig3.add_subplot(111)
    ax3.plot(timeline, p0, color='black', linestyle='--', label='0th Percentile')
    ax3.plot(timeline, p25, color='blue', linestyle='-.', label='25th Percentile')
    ax3.plot(timeline, p50, color='blue', linewidth=2, label='50th Percentile')
    ax3.plot(timeline, p75, color='blue', linestyle='-.', label='75th Percentile')
    ax3.plot(timeline, p90, color='black', linestyle='--', label='90th Percentile')
    ax3.plot(timeline, p98, color='red', linestyle='-', label='98th Percentile')
    ax3.plot(timeline, p100, color='red', linestyle='-', label='100th Percentile')
    ax3.set_xlabel('Simulation Time (s)')
    ax3.set_ylabel('Payload Position Error')
    ax3.set_title('Batched Rollout Payload Position Error Over Time')
    ax3.legend()
    ax3.grid(True)
    plt.savefig('batched_payload_error_over_time.png', dpi=300)
    print("Plot saved: Batched Payload Error Over Time")
    wandb.log({"batched_payload_error_over_time": wandb.Image('batched_payload_error_over_time.png')})
    plt.close(fig3)


def main():
    # Default reward coefficients
    default_reward_coeffs = {
        "distance_reward_coef": 1.0,
        "z_distance_reward_coef": 0.0,
        "velocity_reward_coef": 1.0,
        "safe_distance_coef": 1.0,
        "up_reward_coef": 1.0,
        "linvel_reward_coef": 1.0,
        "ang_vel_reward_coef": 1.0,
        "linvel_quad_reward_coef": 1.0,
        "taut_reward_coef": 1.0,
        "collision_penalty_coef": -1.0,
        "out_of_bounds_penalty_coef": -1.0,
        "smooth_action_coef": -2.0,
        "action_energy_coef": 0.0,
    }
    # Build configuration for IPPO training on multiquad_2x4
    config = {
        "ENV_NAME": "multiquad_2x4",
        "ENV_KWARGS": {
            "reward_coeffs": default_reward_coeffs,
            "obs_noise": 0.0,
            "act_noise": 0.05,
        },
        "TOTAL_TIMESTEPS": 30_000_000,
        "NUM_ENVS": 4096,
        "NUM_STEPS": 512,
        "NUM_MINIBATCHES": 256,
        "UPDATE_EPOCHS": 8,
        "ANNEAL_LR": False,
        "LR":  4e-4,
        "ACTIVATION": "tanh",
        "MAX_GRAD_NORM": 0.5,
        "CLIP_EPS": 0.2,
        "VF_COEF": 0.5,
        "ENT_COEF": 0.01,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "SEED": 0,
        "ACTOR_ARCH": [128, 64, 64, 64],
        "CRITIC_ARCH": [256, 256, 128, 128],
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
    # Merge any sweep overrides into config
    config = {**config, **wandb.config}

    # update nested keys from wandb.config with dot notation
    def _update_nested_config(d, key, value):
        keys = key.split('.')
        target = d
        for subkey in keys[:-1]:
            if subkey not in target or not isinstance(target[subkey], dict):
                target[subkey] = {}
            target = target[subkey]
        target[keys[-1]] = value

    for k, v in wandb.config.items():
        if '.' in k:
            _update_nested_config(config, k, v)

    # terminate if num_steps*num_envs is too large, because of the GPU memory
    if config["NUM_STEPS"] * config["NUM_ENVS"] > 2048*2048:
        raise ValueError("NUM_STEPS * NUM_ENVS is too large. Please reduce them.")

    
    
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
    # Initialize ActorCritic with architectures from config
    network = ActorCritic(
        action_dim=act_dim,
        activation=config["ACTIVATION"],
        actor_arch=config.get("ACTOR_ARCH", [128, 64, 64]),
        critic_arch=config.get("CRITIC_ARCH", [128, 128, 128])
    )
    
    # Define a policy function to map observations to actions using the trained parameters
    def policy_fn(params, obs, key):
        batched_obs = batchify(obs, env.agents, env.num_agents)
        actor_mean = network.apply(params, batched_obs, method=ActorCritic.actor_forward)
        unbatched = unbatchify(actor_mean, env.agents, 1, env.num_agents)
        unbatched = {a: jp.squeeze(val, axis=0) for a, val in unbatched.items()}
        return unbatched
    
   # Simulation: run an episode using the trained policy
    sim_steps = 10000
    rng, rng_sim = jax.random.split(rng)
    state = env.reset(rng_sim)
    rollout = [state[1]]
    
    print("Starting simulation with trained policy...")
    for i in range(sim_steps):
        rng, key = jax.random.split(rng)
        actions = policy_fn(train_state.params, env.get_obs(state[1]), key)  
        rng, key = jax.random.split(rng)
        _, new_state, rewards, dones, info = env.step_env(key, state[1], actions)
        rollout.append(new_state)
        # If any episode terminates, reset the environment and log the new state
        if any(dones.values()):
            rng, reset_key = jax.random.split(rng)
            state = env.reset(reset_key)
            rollout.append(state[1])
        else:
            state = (None, new_state)
    state = jax.block_until_ready(state)
    print("Simulation finished.")
    
    
    # Call the separated video rendering function
    render_video(rollout, env)
    
    def export_to_onnx(module, params, input_shape, onnx_filename, method=None):
        def jax_callable(x):
            return module.apply(params, x, method=method)
        # Use a hardcoded batch size = 1 for export
        save_onnx(jax_callable, [(1, input_shape[1])], onnx_filename)
        print(f"Exported ONNX model: {onnx_filename}")
        return onnx_filename

    # Define input shape for export (use a static batch size)
    input_shape = [1, obs_shape]

    # Use the full parameter tree from train_state
    full_params = train_state.params

    # Export actor and critic using the full parameters with their specific methods.
    actor_onnx = export_to_onnx(
        module=network,
        params=full_params,
        input_shape=input_shape,
        onnx_filename="actor_policy.onnx",
        method=ActorCritic.actor_forward
    )
    critic_onnx = export_to_onnx(
        module=network,
        params=full_params,
        input_shape=input_shape,
        onnx_filename="critic_value.onnx",
        method=ActorCritic.critic_forward
    )
    
    # Log the ONNX models as wandb artifacts.
    actor_artifact = wandb.Artifact("actor_policy", type="model")
    actor_artifact.add_file(actor_onnx)
    critic_artifact = wandb.Artifact("critic_value", type="model")
    critic_artifact.add_file(critic_onnx)
    wandb.log_artifact(actor_artifact)
    wandb.log_artifact(critic_artifact)
    print("ONNX models have been exported and logged to wandb.")
    
    # ---- Call eval_results ----
    def dummy_jit_reset(rng):
        s = env.reset(rng)
        return type("State", (), {"pipeline_state": s[1], "obs": env.get_obs(s[1])})()
    jit_reset = dummy_jit_reset
    jit_inference_fn = lambda obs, key: (policy_fn(train_state.params, obs, key), None)
    def dummy_jit_step(s, ctrl):
        result = env.step_env(jax.random.PRNGKey(0), s.pipeline_state, ctrl)
        new_state = result[1]
        new_obs = env.get_obs(new_state)
        return type("State", (), {"pipeline_state": new_state, "obs": new_obs})()
    jit_step = dummy_jit_step
    eval_results(env, jit_reset, jit_inference_fn, jit_step)
    
if __name__ == "__main__":
    main()