import time
import argparse
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from smax.gridworld.maze import Maze, Actions
from smax.gridworld.ma_maze import MAMaze
from smax.gridworld.grid_viz import GridVisualizer 


def redraw(state, obs, extras):
	extras['viz'].render(extras['params'], state, highlight=False)
	if extras['obs_viz'] is not None:
		extras['obs_viz'].render_grid(np.asarray(obs['image']), k_rot90=3, agent_dir_idx=3)

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
	obs, state, reward, done, info = env.step_env(subkey, extras['state'], jnp.array([action, action]))
	extras['obs'] = obs
	extras['state'] = state
	print(f"reward={reward}, agent_dir={obs['agent_dir']}")

	print(done)

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
	if event.key == 'pageup':
		step(env, Actions.pickup, extras)
		return
	if event.key == 'pagedown':
		step(env, Actions.drop, extras)
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
		default="MAMaze"
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
	args = parser.parse_args()

	if args.env == "Maze":
		env = Maze(
			height=13,
			width=13,
			n_walls=25,
			see_agent=True,
		)
	else:
		env = MAMaze(
			height=13,
			width=13,
			n_walls=25,
			see_agent=True,
			n_agents=2
		)
	params = env.params

	viz = GridVisualizer()
	obs_viz = None
	if args.render_agent_view:
		obs_viz = GridVisualizer()

	with jax.disable_jit(True):
		jit_reset = jax.jit(env.reset_env, static_argnums=(1,))
		# jit_reset = env.reset_env
		key = jax.random.PRNGKey(args.seed)
		key, subkey = jax.random.split(key)
		o0, s0 = jit_reset(subkey)
		viz.render(params, s0, highlight=False)
		if obs_viz is not None:
			obs_viz.render_grid(np.asarray(o0['image']), k_rot90=3, agent_dir_idx=3)

		key, subkey = jax.random.split(key)
		extras = {
			'rng': subkey,
			'state': s0,
			'obs': o0,
			'params':params,
			'viz': viz,
			'obs_viz': obs_viz,
			'jit_reset': jit_reset,
		}

		viz.window.reg_key_handler(partial(key_handler, env, extras))
		viz.show(block=True)

