import copy

import numpy as np
import jax.numpy as jnp
from flax import struct
import chex
import jax


OBJECT_TO_INDEX = {
	"unseen": 0,
	"empty": 1,
	"wall": 2,
	"floor": 3,
	"door": 4,
	"key": 5,
	"ball": 6,
	"box": 7,
	"goal": 8,
	"lava": 9,
	"agent": 10,
}


COLORS = {
    'red'   : np.array([255, 0, 0]),
    'green' : np.array([0, 255, 0]),
    'blue'  : np.array([0, 0, 255]),
    'purple': np.array([112, 39, 195]),
    'yellow': np.array([255, 255, 0]),
    'grey'  : np.array([100, 100, 100]),
}


COLOR_TO_INDEX = {
    'red'   : 0,
    'green' : 1,
    'blue'  : 2,
    'purple': 3,
    'yellow': 4,
    'grey'  : 5,
}


# Map of agent direction indices to vectors
DIR_TO_VEC = jnp.array([
	# Pointing right (positive X)
	(1, 0), # right
	(0, 1), # down
	(-1, 0), # left
	(0, -1), # up
], dtype=jnp.int8)


def make_maze_map(
	params,
	wall_map, 
	goal_pos, 
	agent_pos, 
	agent_dir_idx,
	pad_obs=False):
	# Expand maze map to H x W x C
	empty = jnp.array([OBJECT_TO_INDEX['empty'], 0, 0], dtype=jnp.uint8)
	wall = jnp.array([OBJECT_TO_INDEX['wall'], COLOR_TO_INDEX['grey'], 0], dtype=jnp.uint8)
	maze_map = jnp.array(jnp.expand_dims(wall_map, -1), dtype=jnp.uint8)
	maze_map = jnp.where(maze_map > 0, wall, empty)

	def _get_agent_updates(agent_dir_idx, agent_pos, agent_idx):
		agent = jnp.array([OBJECT_TO_INDEX['agent'], COLOR_TO_INDEX['red']+agent_idx*2, agent_dir_idx], dtype=jnp.uint8)
		agent_x, agent_y = agent_pos
		return agent_x, agent_y, agent

	agent_x_vec, agent_y_vec, agent_vec = jax.vmap(_get_agent_updates, in_axes=(0,0,0))(agent_dir_idx, agent_pos, jnp.arange(params.n_agents))
	maze_map = maze_map.at[agent_y_vec,agent_x_vec,:].set(agent_vec)

	if len(goal_pos.shape) == 1:
		goal = jnp.array([OBJECT_TO_INDEX['goal'], COLOR_TO_INDEX['green'], 0], dtype=jnp.uint8)
		goal_x,goal_y = goal_pos
		maze_map = maze_map.at[goal_y,goal_x,:].set(goal)
	else:
		color_seq = jnp.array(list(GOAL_COLOR_TO_INDEX.values()))[goal_color_sequence]
		goals = jnp.stack(
			[jnp.array([OBJECT_TO_INDEX['goal'], color, 0]) for color in color_seq],
			dtype=jnp.uint8
		).at[:goal_pos.shape[0]].get()
		goal_x = goal_pos.at[:,0].get()
		goal_y = goal_pos.at[:,1].get()
		maze_map = maze_map.at[goal_y,goal_x,:].set(goals)

	# Add observation padding
	if pad_obs:
		padding = params.agent_view_size-1
	else:
		padding = 1

	maze_map_padded = jnp.tile(wall.reshape((1,1,*empty.shape)), (maze_map.shape[0]+2*padding, maze_map.shape[1]+2*padding, 1))
	maze_map_padded = maze_map_padded.at[padding:-padding,padding:-padding,:].set(maze_map)

	# Add surrounding walls
	wall_start = padding-1 # start index for walls
	wall_end_y = maze_map_padded.shape[0] - wall_start - 1
	wall_end_x = maze_map_padded.shape[1] - wall_start - 1
	maze_map_padded = maze_map_padded.at[wall_start,wall_start:wall_end_x+1,:].set(wall) # top
	maze_map_padded = maze_map_padded.at[wall_end_y,wall_start:wall_end_x+1,:].set(wall) # bottom
	maze_map_padded = maze_map_padded.at[wall_start:wall_end_y+1,wall_start,:].set(wall) # left
	maze_map_padded = maze_map_padded.at[wall_start:wall_end_y+1,wall_end_x,:].set(wall) # right

	return maze_map_padded
