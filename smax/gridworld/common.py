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
	"onion": 3,
	"onion_pile": 4,
	"plate": 5,
	"plate_pile": 6,
	"goal": 7,
	"pot": 8,
	"dish": 9,
	"agent": 10,
}


COLORS = {
    'red'   : np.array([255, 0, 0]),
    'green' : np.array([0, 255, 0]),
    'blue'  : np.array([0, 0, 255]),
    'purple': np.array([112, 39, 195]),
    'yellow': np.array([255, 255, 0]),
    'grey'  : np.array([100, 100, 100]),
    'white' : np.array([255, 255, 255]),
	'black' : np.array([25, 25, 25]),
	'orange': np.array([230, 180, 0]),
}


COLOR_TO_INDEX = {
    'red'   : 0,
    'green' : 1,
    'blue'  : 2,
    'purple': 3,
    'yellow': 4,
    'grey'  : 5,
    'white' : 6,
	'black' : 7,
}

OBJECT_INDEX_TO_VEC = jnp.array([
	jnp.array([OBJECT_TO_INDEX['unseen'], 0, 0], dtype=jnp.uint8),
	jnp.array([OBJECT_TO_INDEX['empty'], 0, 0], dtype=jnp.uint8),
	jnp.array([OBJECT_TO_INDEX['wall'], COLOR_TO_INDEX['grey'], 0], dtype=jnp.uint8),
	jnp.array([OBJECT_TO_INDEX['onion'], COLOR_TO_INDEX["yellow"], 0], dtype=jnp.uint8),
	jnp.array([OBJECT_TO_INDEX['onion_pile'], COLOR_TO_INDEX["yellow"], 0], dtype=jnp.uint8),
	jnp.array([OBJECT_TO_INDEX['plate'], COLOR_TO_INDEX["white"], 0], dtype=jnp.uint8),
	jnp.array([OBJECT_TO_INDEX['plate_pile'], COLOR_TO_INDEX["white"], 0], dtype=jnp.uint8),
	jnp.array([OBJECT_TO_INDEX['goal'], COLOR_TO_INDEX['green'], 0], dtype=jnp.uint8),
	jnp.array([OBJECT_TO_INDEX['pot'], COLOR_TO_INDEX['black'], 0], dtype=jnp.uint8),
	jnp.array([OBJECT_TO_INDEX['dish'], COLOR_TO_INDEX["white"], 0], dtype=jnp.uint8),
	jnp.array([OBJECT_TO_INDEX['agent'], COLOR_TO_INDEX['red'], 0]),  					# Default color and direction
])

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


def make_overcooked_map(
	params,
	wall_map,
	goal_pos,
	agent_pos,
	agent_dir_idx,
	plate_pile_pos,
	onion_pile_pos,
	pot_pos,
	pad_obs=False):

	# Expand maze map to H x W x C
	empty = jnp.array([OBJECT_TO_INDEX['empty'], 0, 0], dtype=jnp.uint8)
	wall = jnp.array([OBJECT_TO_INDEX['wall'], COLOR_TO_INDEX['grey'], 0], dtype=jnp.uint8)
	maze_map = jnp.array(jnp.expand_dims(wall_map, -1), dtype=jnp.uint8)
	maze_map = jnp.where(maze_map > 0, wall, empty)

	# Add agents
	def _get_agent_updates(agent_dir_idx, agent_pos, agent_idx):
		agent = jnp.array([OBJECT_TO_INDEX['agent'], COLOR_TO_INDEX['red']+agent_idx*2, agent_dir_idx], dtype=jnp.uint8)
		agent_x, agent_y = agent_pos
		return agent_x, agent_y, agent

	agent_x_vec, agent_y_vec, agent_vec = jax.vmap(_get_agent_updates, in_axes=(0,0,0))(agent_dir_idx, agent_pos, jnp.arange(params.n_agents))
	maze_map = maze_map.at[agent_y_vec,agent_x_vec,:].set(agent_vec)

	# Add goals
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

	# Add onions
	onion_x = onion_pile_pos[:,0]
	onion_y = onion_pile_pos[:,1]
	onions = jnp.stack(
			[jnp.array([OBJECT_TO_INDEX['onion_pile'], COLOR_TO_INDEX["yellow"], 0]) for _ in range(10)],
			dtype=jnp.uint8
		).at[:onion_pile_pos.shape[0]].get()
	maze_map = maze_map.at[onion_y, onion_x, :].set(onions)

	# Add plates
	plate_x = plate_pile_pos[:, 0]
	plate_y = plate_pile_pos[:, 1]
	plates = jnp.stack(
		[jnp.array([OBJECT_TO_INDEX['plate_pile'], COLOR_TO_INDEX["white"], 0]) for _ in range(10)],
		dtype=jnp.uint8
	).at[:plate_pile_pos.shape[0]].get()
	maze_map = maze_map.at[plate_y, plate_x, :].set(plates)

	# Add pots
	pot_x = pot_pos[:, 0]
	pot_y = pot_pos[:, 1]
	pots = jnp.stack(
		[jnp.array([OBJECT_TO_INDEX['pot'], COLOR_TO_INDEX["black"], 23]) for _ in range(10)],
		dtype=jnp.uint8
	).at[:pot_pos.shape[0]].get()
	maze_map = maze_map.at[pot_y, pot_x, :].set(pots)

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
