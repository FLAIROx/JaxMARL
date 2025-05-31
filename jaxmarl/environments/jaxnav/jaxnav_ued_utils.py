import jax
import jax.numpy as jnp
import chex
# from .level import Level
# from .env import DIR_TO_VEC
from .jaxnav_env import State
from .maps import GridMapPolygonAgents
from enum import IntEnum
import numpy as np
from typing import Callable
from functools import partial



""" 
Mutation strategy:
1. flip wall: compute free cells, choose one at random, flip it to a wall
- should probably switch to a MapData class rather than just an array
  as it would be good to store free cells. Ignore for now

"""    
    

def make_level_mutator(max_num_edits: int, map: GridMapPolygonAgents):
    
    class Mutations(IntEnum):
        # Turn left, turn right, move forward
        NO_OP = 0
        FLIP_WALL = 1
        MOVE_GOAL = 2
        
    
    def move_goal_flip_walls(rng, state: State, n: int = 1):
        def _mutate(carry, step):
            state = carry
            rng, mutation = step

            def _apply(rng, state):    
                rng, arng, brng = jax.random.split(rng, 3)

                is_flip_wall = jnp.equal(mutation, Mutations.FLIP_WALL.value)
                mutated_state = flip_wall(arng, map, state)
                next_state = jax.tree.map(lambda x,y: jax.lax.select(is_flip_wall, x, y), mutated_state, state)

                is_move_goal = jnp.equal(mutation, Mutations.MOVE_GOAL.value)
                mutated_state = move_goal(brng, map, state)
                next_state = jax.tree.map(lambda x,y: jax.lax.select(is_move_goal, x, y), mutated_state, next_state)
                
                return next_state
                
            return jax.lax.cond(mutation != -1, _apply, lambda *_: state, rng, state), None
        
        
        rng, nrng, *mrngs = jax.random.split(rng, max_num_edits+2)
        mutations = jax.random.choice(nrng, np.arange(len(Mutations)), (max_num_edits,))
        mutations = jnp.where(jnp.arange(max_num_edits) < n, mutations, -1) # mask out extra mutations

        new_level, _ = jax.lax.scan(_mutate, state, (jnp.array(mrngs), mutations))

        return new_level
    
    return move_goal_flip_walls
    
def flip_wall(rng, map: GridMapPolygonAgents, state: State):
    wall_map = state.map_data
    h,w = wall_map.shape
        
    # goal_map_mask = jnp.any(jax.vmap(
    #     map.get_circle_map_occupancy_mask,
    #     in_axes=(0, None, None)
    # )(state.goal, wall_map, 0.3), axis=0)
    
    goal_map_mask = wall_map

    start_map_mask = jnp.any(jax.vmap(
        map.get_agent_map_occupancy_mask, in_axes=(0,0,None)
    )(state.pos, state.theta, wall_map), axis=0)
    

    goal = jnp.floor(state.goal).astype(jnp.int32)
    goal_map_mask = goal_map_mask.at[goal[:, 1], goal[:, 0]].set(1)

    map_mask = start_map_mask | map._gen_base_grid() | goal_map_mask
    

    flip_idx = jax.random.choice(rng, np.arange(h*w), p=jnp.logical_not(map_mask.flatten()))

    flip_y = flip_idx // w
    flip_x = flip_idx % w
    
    flip_val = 1-wall_map.at[flip_y, flip_x].get()
    next_wall_map = wall_map.at[flip_y, flip_x].set(flip_val)
    print('next_wall_map', next_wall_map)
    return state.replace(map_data=next_wall_map)


def move_goal(rng, map:GridMapPolygonAgents, state:State):
    wall_map = state.map_data
    h,w = wall_map.shape
    
    rng, _rng = jax.random.split(rng)
    agent_idx = jax.random.choice(_rng, np.arange(state.pos.shape[0]))
    
    # goal_map_masks = jax.vmap(
    #     map.get_circle_map_occupancy_mask,
    #     in_axes=(0, None, None)
    # )(state.goal, wall_map, 1.0)
    goal_map_masks = jnp.repeat(wall_map[None], state.pos.shape[0], axis=0)
    # goal_map_masks = goal_map_masks.at[agent_idx].set(jnp.zeros(wall_map.shape, dtype=jnp.int32))
    goal_map_mask = jnp.any(goal_map_masks, axis=0)
    current_goal = jnp.floor(state.goal[agent_idx]).astype(jnp.int32)
    goal_map_mask = goal_map_mask.at[current_goal[1], current_goal[0]].set(0)
    
    start_map_mask = jnp.any(jax.vmap(
        map.get_agent_map_occupancy_mask, in_axes=(0,0,None)
    )(state.pos, state.theta, wall_map), axis=0)
    
    map_mask = start_map_mask | map._gen_base_grid() | goal_map_mask
    
    next_idx = jax.random.choice(rng, np.arange(h*w), p=jnp.logical_not(map_mask.flatten()))
    next_goal_y = next_idx // w
    next_goal_x = next_idx %  w
    
    goals = state.goal.at[agent_idx].set(jnp.array([next_goal_x, next_goal_y]) + 0.5) # Make the goal in the center of the cell.
    
    return state.replace(goal=goals)
    
    
    
    