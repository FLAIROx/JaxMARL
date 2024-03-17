import argparse
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from jaxmarl.gridworld.maze import Maze #, Actions
from jaxmarl.gridworld.ma_maze import MAMaze
from jaxmarl.environments.overcooked.overcooked import Overcooked
from jaxmarl.environments.overcooked.layouts import layouts


def redraw(state, obs, extras):
    extras['viz'].render(extras['params'], state, highlight=False)

    if extras['obs_viz'] is not None:
        if extras['env'] == "MAMaze" or "Overcooked":
            obs_viz.render_grid(np.asarray(obs['image'][0]), k_rot90=3, agent_dir_idx=[3])
            obs_viz2.render_grid(np.asarray(obs['image'][1]), k_rot90=3, agent_dir_idx=[3])
        else:
            obs_viz.render_grid(np.asarray(obs['image']), k_rot90=3, agent_dir_idx=3)


def reset(key, env, extras):
    key, subkey = jax.random.split(extras['rng'])
    obs, state = extras['jit_reset'](subkey)

    extras['rng'] = key
    extras['obs'] = obs
    extras['state'] = state

    redraw(state, obs, extras)


def step(env, action, extras):
    # TODO: Handle actions better (e.g. choose which agent to control)
    key, subkey = jax.random.split(extras['rng'])

    print("action:", jnp.array([action, action.left]))
    obs, state, reward, done, info = jax.jit(env.step_env)(subkey, extras['state'], jnp.array([action, action]))
    extras['obs'] = obs
    extras['state'] = state
    print(f"reward={reward}, agent_dir={obs['agent_dir']}, agent_inv={state.agent_inv}")
    
    if extras["debug"]:
        layers = [f"player_{i}_loc" for i in range(2)]
        layers.extend([f"player_{i // 4}_orientation_{i % 4}" for i in range(8)])
        layers.extend([
            "pot_loc",
            "counter_loc",
            "onion_disp_loc",
            "tomato_disp_loc",
            "plate_disp_loc",
            "serve_loc",
            "onions_in_pot",
            "tomatoes_in_pot",
            "onions_in_soup",
            "tomatoes_in_soup",
            "soup_cook_time_remaining",
            "soup_done",
            "plates",
            "onions",
            "tomatoes",
            "urgency"
        ])
        print("obs_shape: ", obs["image"].shape)
        print("OBS: \n", obs["image"][1])
        debug_obs = jnp.transpose(obs["image"][1], (2,0,1))
        for i, layer in enumerate(layers):
            print(layer)
            print(debug_obs[i])
    # print(f"agent obs =\n {obs}")

    if done or (jnp.array([action, action]) == Actions.done).any():
        key, subkey = jax.random.split(subkey)
        reset(subkey, env, extras)
    else:
        redraw(state, obs, extras)

    extras['rng'] = key


def key_handler(env, extras, event):
    print('pressed', event.key)

    if event.key == 'escape':
        window.close()
        return

    if event.key == 'backspace':
        extras['jit_reset']((env, extras))
        return

    if event.key == 'left':
        step(env, Actions.left, extras)
        return
    if event.key == 'right':
        step(env, Actions.right, extras)
        return
    if event.key == 'up':
        step(env, Actions.forward, extras)
        return

    # Spacebar
    if event.key == ' ':
        step(env, Actions.toggle, extras)
        return
    if event.key == '[':
        step(env, Actions.pickup, extras)
        return
    if event.key == ']':
        step(env, Actions.drop, extras)
        return

    if event.key == 'enter':
        step(env, Actions.done, extras)
        return

def key_handler_overcooked(env, extras, event):
    print('pressed', event.key)

    if event.key == 'escape':
        window.close()
        return
    if event.key == 'backspace':
        extras['jit_reset']((env, extras))
        return

    if event.key == 'left':
        step(env, Actions.left, extras)
        return
    if event.key == 'right':
        step(env, Actions.right, extras)
        return
    if event.key == 'up':
        # step(env, Actions.forward, extras)
        step(env, Actions.up, extras)
        return
    if event.key == 'down':
        step(env, Actions.down, extras)
        return

    # Spacebar
    if event.key == ' ':
        step(env, Actions.interact, extras)
        return
    if event.key == 'tab':
        step(env, Actions.stay, extras)
        return
    if event.key == 'enter':
        step(env, Actions.done, extras)
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        type=str,
        help="Environment name",
        default="Overcooked"
    )
    parser.add_argument(
        "--layout",
        type=str,
        help="Overcooked layout",
        default=""
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed to generate the environment with",
        default=0
    )
    parser.add_argument(
        '--render_agent_view',
        default=False,
        help="draw the agent sees (partially observable view)",
        action='store_true'
    )
    parser.add_argument(
        '--height',
        default=13,
        type=int,
        help="height",
    )
    parser.add_argument(
        '--width',
        default=13,
        type=int,
        help="width",
    )
    parser.add_argument(
        '--n_walls',
        default=50,
        type=int,
        help="Number of walls",
    )
    parser.add_argument(
        '--agent_view_size',
        default=5,
        type=int,
        help="Number of walls",
    )
    parser.add_argument(
        '--debug',
        default=False,
        help="Debug mode",
        action='store_true'
    )
    args = parser.parse_args()

    if args.env == "Maze":
        env = Maze(
            height=13,
            width=13,
            n_walls=25,
            see_agent=True,
        )
        from jaxmarl.gridworld.grid_viz import GridVisualizer as Visualizer
        from jaxmarl.gridworld.maze import Actions

    elif args.env == "MAMaze":
        env = MAMaze(
            height=13,
            width=13,
            n_walls=25,
            see_agent=True,
            n_agents=2
        )
        from jaxmarl.gridworld.grid_viz import GridVisualizer as Visualizer
        from jaxmarl.gridworld.maze import Actions

    elif args.env == "Overcooked":
        if len(args.layout) > 0:
            layout = layouts[args.layout]
            env = Overcooked(
                height=layout["height"],
                width=layout["width"],
                n_walls=1,
                see_agent=True,
                n_agents=2,
                fixed_layout=True,
                layout=layout
            )
        else:
            env = Overcooked(
                height=13,
                width=13,
                n_walls=25,
                see_agent=True,
                n_agents=2
            )
        from jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer as Visualizer
        from jaxmarl.environments.overcooked.overcooked import Actions

    params = env.params

    viz = Visualizer()
    obs_viz = None
    obs_viz2 = None
    if args.render_agent_view:
        obs_viz = Visualizer()
        if args.env == "MAMaze" or "Overcooked":
            obs_viz2 = Visualizer()

    with jax.disable_jit(True):
        jit_reset = jax.jit(env.reset_env, static_argnums=(1,))
        # jit_reset = env.reset_env
        key = jax.random.PRNGKey(args.seed)
        key, subkey = jax.random.split(key)
        o0, s0 = jit_reset(subkey)
        viz.render(params, s0, highlight=False)
        if obs_viz is not None:
            if args.env == "MAMaze" or args.env == "Overcooked":
                obs_viz.render_grid(np.asarray(o0['image'][0]), k_rot90=3, agent_dir_idx=[3])
                obs_viz2.render_grid(np.asarray(o0['image'][1]), k_rot90=3, agent_dir_idx=[3])
            else:
                obs_viz.render_grid(np.asarray(o0['image']), k_rot90=3, agent_dir_idx=3)

        key, subkey = jax.random.split(key)
        extras = {
            'rng': subkey,
            'state': s0,
            'obs': o0,
            'params': params,
            'viz': viz,
            'obs_viz': obs_viz,
            'obs_viz2': obs_viz2,
            'jit_reset': jit_reset,
            'env': args.env,
            'debug': args.debug
        }

        if args.env == "Overcooked":
            viz.window.reg_key_handler(partial(key_handler_overcooked, env, extras))
            viz.show(block=True)
        else:
            viz.window.reg_key_handler(partial(key_handler, env, extras))
            viz.show(block=True)
