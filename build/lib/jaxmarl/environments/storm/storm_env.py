import math
from enum import IntEnum
from typing import Any, Optional, Tuple, Union, List, Dict
import itertools

import chex
import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
import numpy as onp

from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.environments import spaces
from flax.struct import dataclass

from .rendering import (
    downsample,
    fill_coords,
    highlight_img,
    point_in_circle,
    point_in_rect,
    point_in_triangle,
    rotate_fn,
)

GRID_SIZE = 8
OBS_SIZE = 5
PADDING = OBS_SIZE - 1
NUM_TYPES = 5  # empty (0), red (1), blue, red coin, blue coin, wall, interact
NUM_COINS = 6  # per type
NUM_COIN_TYPES = 2
NUM_OBJECTS = (
    2 + NUM_COIN_TYPES * NUM_COINS + 1
)  # red, blue, 2 red coin, 2 blue coin

INTERACT_THRESHOLD = 0


@dataclass
class State:
    # n players x 2
    agent_positions: List[jnp.ndarray]
    inner_t: int
    outer_t: int
    grid: jnp.ndarray
    # n players x 2
    agent_inventories: List[jnp.ndarray]
    # coin positions
    coin_coop: List[jnp.ndarray]
    coin_defect: List[jnp.ndarray]
    agent_freezes: List[int]
    coin_timer_coop: List[int]
    coin_timer_defect: List[int]


class Actions(IntEnum):
    left = 0
    right = 1
    forward = 2
    interact = 3
    stay = 4


class Items(IntEnum):
    empty = 0
    agent1 = 1
    agent2 = 2
    agent3 = 3
    agent4 = 4
    agent5 = 5
    agent6 = 6
    agent7 = 7
    agent8 = 8
    cooperation_coin = 9
    defection_coin = 10
    wall = 11
    interact = 12
    #agents if  1 < val < item.WALL 

AGENT_IDX = jnp.array([1,2,3,4,5,6,7,8])
ROTATIONS = jnp.array(
    [
        [0, 0, 1],  # turn left
        [0, 0, -1],  # turn right
        [0, 0, 0],  # forward
        [0, 0, 0],  # stay
        [0, 0, 0],  # zap`
    ],
    dtype=jnp.int8,
)

STEP = jnp.array(
    [
        [0, 1, 0],  # up
        [1, 0, 0],  # right
        [0, -1, 0],  # down
        [-1, 0, 0],  # left
    ],
    dtype=jnp.int8,
)

GRID = jnp.zeros(
    (GRID_SIZE + 2 * PADDING, GRID_SIZE + 2 * PADDING),
    dtype=jnp.int8,
)

# First layer of Padding is Wall
GRID = GRID.at[PADDING - 1, :].set(5)
GRID = GRID.at[GRID_SIZE + PADDING, :].set(5)
GRID = GRID.at[:, PADDING - 1].set(5)
GRID = GRID.at[:, GRID_SIZE + PADDING].set(5)

COIN_SPAWNS = [
    [1, 1],
    [1, 2],
    [2, 1],
    [1, GRID_SIZE - 2],
    [2, GRID_SIZE - 2],
    [1, GRID_SIZE - 3],
    # [2, 2],
    # [2, GRID_SIZE - 3],
    [GRID_SIZE - 2, 2],
    [GRID_SIZE - 3, 1],
    [GRID_SIZE - 2, 1],
    [GRID_SIZE - 2, GRID_SIZE - 2],
    [GRID_SIZE - 2, GRID_SIZE - 3],
    [GRID_SIZE - 3, GRID_SIZE - 2],
    # [GRID_SIZE - 3, 2],
    # [GRID_SIZE - 3, GRID_SIZE - 3],
]

COIN_SPAWNS = jnp.array(
    COIN_SPAWNS,
    dtype=jnp.int8,
)

RED_SPAWN = jnp.array(
    COIN_SPAWNS[::2, :],
    dtype=jnp.int8,
)

BLUE_SPAWN = jnp.array(
    COIN_SPAWNS[1::2, :],
    dtype=jnp.int8,
)

AGENT_SPAWNS = [
    [0, 0],
    [0, 1],
    [0, 2],
    [1, 0],
    # [1, 1],
    [1, 2],
    [0, GRID_SIZE - 1],
    [0, GRID_SIZE - 2],
    [0, GRID_SIZE - 3],
    [1, GRID_SIZE - 1],
    # [1, GRID_SIZE - 2],
    # [1, GRID_SIZE - 3],
    [GRID_SIZE - 1, 0],
    [GRID_SIZE - 1, 1],
    [GRID_SIZE - 1, 2],
    [GRID_SIZE - 2, 0],
    # [GRID_SIZE - 2, 1],
    # [GRID_SIZE - 2, 2],
    [GRID_SIZE - 1, GRID_SIZE - 1],
    [GRID_SIZE - 1, GRID_SIZE - 2],
    [GRID_SIZE - 1, GRID_SIZE - 3],
    [GRID_SIZE - 2, GRID_SIZE - 1],
    # [GRID_SIZE - 2, GRID_SIZE - 2],
    [GRID_SIZE - 2, GRID_SIZE - 3],
]

# Get all combinations of AGENT_SPAWNS with length 8
agent_spawn_combinations = list(itertools.combinations(AGENT_SPAWNS, 8))

# Convert the list of combinations to a JAX numpy array
AGENT_SPAWNS = jnp.array(agent_spawn_combinations, dtype=jnp.int8)


PLAYER1_COLOUR = (255.0, 127.0, 14.0) #kinda orange
PLAYER2_COLOUR = (31.0, 119.0, 180.0) #kinda blue
PLAYER3_COLOUR = (236.0, 64.0, 122.0) #kinda pink
PLAYER4_COLOUR = (255.0, 235.0, 59.0) #yellow
PLAYER5_COLOUR = (41.0, 182.0, 246.0) #baby blue
PLAYER6_COLOUR = (171.0, 71.0, 188.0) #purple
PLAYER7_COLOUR = (121.0, 85.0, 72.0) #brown
PLAYER8_COLOUR = (255.0, 205.0, 210.0) #salmon
GREEN_COLOUR = (44.0, 160.0, 44.0)
RED_COLOUR = (214.0, 39.0, 40.0)



class InTheGrid(MultiAgentEnv):
    """
    JAX Compatible version of *inTheMatix environment.
    """

    # used for caching
    tile_cache: Dict[Tuple[Any, ...], Any] = {}

    def __init__(
        self,
        num_inner_steps=152,
        num_outer_steps=1,
        num_agents=2,
        fixed_coin_location=True,
        payoff_matrix=jnp.array([[[3, 0], [5, 1]], [[3, 5], [0, 1]]]),
        freeze_penalty=5,
    ):

        super().__init__(num_agents=num_agents)
        self.agents = list(range(num_agents))

        def _get_obs_point(x: int, y: int, dir: int) -> jnp.ndarray:
            x, y = x + PADDING, y + PADDING
            x = jnp.where(dir == 0, x - (OBS_SIZE // 2), x)
            x = jnp.where(dir == 2, x - (OBS_SIZE // 2), x)
            x = jnp.where(dir == 3, x - (OBS_SIZE - 1), x)

            y = jnp.where(dir == 1, y - (OBS_SIZE // 2), y)
            y = jnp.where(dir == 2, y - (OBS_SIZE - 1), y)
            y = jnp.where(dir == 3, y - (OBS_SIZE // 2), y)
            return x, y

        def _get_obs(state: State) -> jnp.ndarray:
            # create state
            grid = jnp.pad(
                state.grid,
                ((PADDING, PADDING), (PADDING, PADDING)),
                constant_values=Items.wall,
            )
            def mini_obs(agent_pos, agent_idx, other_agents_idx, other_agents_pos):
                x, y = _get_obs_point(
                agent_pos[0], agent_pos[1], agent_pos[2]
                )
                grid1 = jax.lax.dynamic_slice(
                    grid,
                    start_indices=(x, y),
                    slice_sizes=(OBS_SIZE, OBS_SIZE),
                )
                # rotate
                grid1 = jnp.where(
                    agent_pos[2] == 1,
                    jnp.rot90(grid1, k=1, axes=(0, 1)),
                    grid1,
                )
                grid1 = jnp.where(
                    agent_pos[2] == 2,
                    jnp.rot90(grid1, k=2, axes=(0, 1)),
                    grid1,
                )
                grid1 = jnp.where(
                    agent_pos[2] == 3,
                    jnp.rot90(grid1, k=3, axes=(0, 1)),
                    grid1,
                )
                # angle1 = -1 * jnp.ones_like(grid1, dtype=jnp.int8)
            #      i look where the blue agent is (with its own orientation)
            #     i correct its orientation relative to mine
            #     angle is 

                def check_angle(grid1, item, other_agent_pos):    
                    angle1 = jnp.where(
                        grid1 == item,
                        ((other_agent_pos[2] - agent_pos[2])+1) % 5,
                        0,
                    )
                    return angle1
                
                vmap_check_angle = jax.vmap(check_angle, (None, 0, 0), 0)
                angle1 = vmap_check_angle(grid1, other_agents_idx, other_agents_pos)
                angle1 = jnp.sum(angle1, axis=0) - 1
                angle1 = jax.nn.one_hot(angle1, 4)
                # one-hot (drop first channel as its empty blocks)
                grid1 = jax.nn.one_hot(grid1 - 1, len(Items) - 1, dtype=jnp.int8)
                _grid1 = grid1.at[:, :, 0].set(grid1[:, :, agent_idx])
                _grid1 = _grid1.at[:, :, agent_idx].set(grid1[:, :, 0])
                obs1 = jnp.concatenate([_grid1, angle1], axis=-1)
                return obs1

            vmap_mini_obs = jax.vmap(mini_obs, (0, 0, 0, 0), 0)
            agent_indices = jnp.arange(1, num_agents+1)
            expanded_indices = agent_indices - 1
            expanded_indices = expanded_indices[:, jnp.newaxis]
            # Create a matrix with the desired shape and fill it with the expanded_indices
            other_agent_indices = expanded_indices + jnp.arange(1, num_agents)
            # Correct the indices by adding 1 where the agent's own index is excluded
            other_agent_indices = other_agent_indices + (other_agent_indices >= expanded_indices)

            # Create an array of agent indices
            agent_indices = jnp.arange(num_agents)

            # Add a new axis to agent_indices and agent_positions
            agent_indices_expanded = agent_indices[:, jnp.newaxis]
            agent_positions_expanded = state.agent_positions[:, jnp.newaxis]

            # Create a matrix of other agent indices for each agent
            other_agent_indices = agent_indices_expanded + jnp.arange(1, num_agents)
            other_agent_indices = other_agent_indices + (other_agent_indices >= agent_indices_expanded)

            # Use other_agent_indices to index into agent_positions
            other_agent_positions = jnp.take(state.agent_positions, other_agent_indices, axis=0)

            obs = vmap_mini_obs(state.agent_positions, agent_indices, other_agent_indices, other_agent_positions)
            pickups = jnp.sum(state.agent_inventories, axis=1) > INTERACT_THRESHOLD
            def agent2show(inventory, freeze):
                agent_to_show = jnp.where(
                    freeze >= 0, inventory, 0
                )
                return agent_to_show
            vmap_agent2show = jax.vmap(agent2show, (0, 0), (0))
            agents2show = vmap_agent2show(state.agent_inventories, state.agent_freezes)
            return {
                "observations": obs,
                "inventory": jnp.array(
                    [
                        state.agent_inventories[:, 0],
                        state.agent_inventories[:, 1],
                        pickups,
                        agents2show[:, 0],
                        agents2show[:, 1],
                    ],
                    dtype=jnp.int8,
                ),
            }

        def _get_reward(state, pair) -> jnp.ndarray:
            inv1 = state.agent_inventories[pair[0]] / state.agent_inventories[pair[0]].sum()
            inv2 = state.agent_inventories[pair[1]] / state.agent_inventories[pair[1]].sum()
            r1 = inv1 @ payoff_matrix[0] @ inv2.T
            r2 = inv1 @ payoff_matrix[1] @ inv2.T
            return jnp.array([r1, r2])


        def _interact(state: State, actions: jnp.array, rng_key: jnp.ndarray) -> Tuple[jnp.ndarray, State]:
            interact_idx = jnp.int8(Items.interact)

            # Remove old interacts from the grid
            grid = jnp.where(
                state.grid == interact_idx, jnp.int8(Items.empty), state.grid
            )
            state = state.replace(grid=grid)

            # Get agent positions and orientations
            agent_positions = state.agent_positions[:, :2]
            agent_orientations = state.agent_positions[:, 2]

            zaps = actions == Actions.interact
            # print(zaps, 'zaps')
            def get_target(agent_pos, grid):
                target = jnp.clip(
                    agent_pos + STEP[agent_pos[2]], 0, GRID_SIZE - 1
                )
                interact = (
                    jnp.isin(grid[target[0], target[1]], AGENT_IDX)
                )
                target_ahead = jnp.clip(
                    agent_pos + 2 * STEP[agent_pos[2]], 0, GRID_SIZE - 1
                )
                interact_ahead = (
                    jnp.isin(grid[target_ahead[0], target_ahead[1]], AGENT_IDX)
                )
                target_right = (
                    agent_pos
                    + STEP[agent_pos[2]]
                    + STEP[(agent_pos[2] + 1) % 4]
                )  
                oob = jnp.logical_or(
                    (target_right > GRID_SIZE - 1).any(),
                    (target_right < 0).any(),
                )
                target_right = jnp.where(oob, target, target_right)

                interact_right = (
                jnp.isin(grid[target_right[0], target_right[1]], AGENT_IDX)
                )
                target_left = (
                    agent_pos
                    + STEP[agent_pos[2]]
                    + STEP[(agent_pos[2] - 1) % 4]
                )
                oob = jnp.logical_or(
                    (target_left > GRID_SIZE - 1).any(),
                    (target_left < 0).any(),
                )
                target_left = jnp.where(oob, target, target_left)
                interact_left = (
                jnp.isin(grid[target_left[0], target_left[1]], AGENT_IDX)
                )
                interact = jnp.logical_or(
                    interact,
                    jnp.logical_or(
                        interact_ahead,
                        jnp.logical_or(interact_right, interact_left),
                    ),
                )
                interact_targets = jnp.array([grid[target[0], target[1]],
                                              grid[target_ahead[0], target_ahead[1]],
                                              grid[target_right[0], target_right[1]],
                                              grid[target_left[0], target_left[1]]])
                return interact, target, target_ahead, target_right, target_left, interact_targets
            
            vmap_get_target = jax.vmap(get_target, (0, None), (0, 0, 0, 0, 0, 0))
            agent_positions, target, target_ahead, target_right, target_left, interact_targets = vmap_get_target(state.agent_positions, state.grid)
            # print(target, target_ahead, target_right, target_left, interact_targets,'target, target_ahead, target_right, target_left, interact_targets')

            def update_grid_interact(i, val):
                grid, target, target_ahead, target_right, target_left, zaps = val
                t_i, t_a_i, t_r_i, t_l_i = target[i], target_ahead[i], target_right[i], target_left[i]
                aux_grid = jnp.copy(grid)
                item = jnp.where(
                    grid[t_i[0], t_i[1]],
                    grid[t_i[0], t_i[1]],
                    interact_idx,
                )
                aux_grid = aux_grid.at[t_i[0], t_i[1]].set(item)

                item = jnp.where(
                    grid[t_a_i[0], t_a_i[1]],
                    grid[t_a_i[0], t_a_i[1]],
                    interact_idx,
                )
                aux_grid = aux_grid.at[
                    t_a_i[0], t_a_i[1]
                ].set(item)

                item = jnp.where(
                    grid[t_r_i[0], t_r_i[1]],
                    grid[t_r_i[0], t_r_i[1]],
                    interact_idx,
                )
                aux_grid = aux_grid.at[
                    t_r_i[0], t_r_i[1]
                ].set(item)

                item = jnp.where(
                    grid[t_l_i[0], t_l_i[1]],
                    grid[t_l_i[0], t_l_i[1]],
                    interact_idx,
                )
                aux_grid = aux_grid.at[t_l_i[0], t_l_i[1]].set(
                    item
                )
                grid = jnp.where(zaps[i], aux_grid, grid)
                return grid, target, target_ahead, target_right, target_left, zaps

            grid, _, _, _, _, _ = jax.lax.fori_loop(0, num_agents, update_grid_interact, (state.grid, target, target_ahead, target_right, target_left, zaps))
            state = state.replace(grid=grid)
            # Create a function to generate agent_pairs without itertools
            def generate_agent_pairs(num_agents):
                pairs = jnp.array([(i, j) for i in range(num_agents) for j in range(i + 1, num_agents)])
                return pairs

            agent_pairs = jnp.array(generate_agent_pairs(num_agents))
            # print(agent_pairs.shape, 'agent pairs')

            # Sort the agent pairs randomly using the provided rng_key
            random_key, subkey = jax.random.split(rng_key)
            shuffled_indices = jax.random.permutation(subkey, jnp.arange(agent_pairs.shape[0]))

            # Create a boolean array to track whether each agent is in an interaction
            agent_interaction_mask = jnp.zeros((num_agents,), dtype=bool)
            pairwise_mask = jnp.zeros((agent_pairs.shape[0],), dtype=bool)

            def process_agent_interactions(i, val):
                agent_interaction_mask, shuffled_indices, interact_targets, agent_inventories, pairwise_mask, zaps = val
                pair_idx = shuffled_indices[i]
                agent0, agent1 = agent_pairs[pair_idx]
                agent1_in_agent0 = jnp.isin(agent1+1, interact_targets[agent0])
                agent0_in_agent1 = jnp.isin(agent0+1, interact_targets[agent1])
                pickup0 = agent_inventories[agent0].sum() > INTERACT_THRESHOLD
                pickup1 = agent_inventories[agent1].sum() > INTERACT_THRESHOLD
                is_interacting = jnp.logical_and(agent1_in_agent0, agent0_in_agent1)
                is_interacting = jnp.logical_and(is_interacting, jnp.logical_and(pickup0, pickup1))
                is_interacting = jnp.logical_and(is_interacting, zaps[agent0])
                is_interacting = jnp.logical_and(is_interacting, zaps[agent1])
                interacting = jnp.logical_and(is_interacting, jnp.logical_not(agent_interaction_mask[agent0] | agent_interaction_mask[agent1]))
                interacting0 = jnp.logical_or(is_interacting, agent_interaction_mask[agent0])
                interacting1 = jnp.logical_or(is_interacting, agent_interaction_mask[agent1])
                agent_interaction_mask = agent_interaction_mask.at[agent0].set(interacting0)
                agent_interaction_mask = agent_interaction_mask.at[agent1].set(interacting1)
                pairwise_mask = pairwise_mask.at[pair_idx].set(interacting)

                return (agent_interaction_mask,  shuffled_indices, interact_targets, agent_inventories, pairwise_mask, zaps)
            
            # for agent0,agent1 in [(0,1),(0,2),(1,2)]:
            #     agent1_in_agent0 = jnp.isin(agent1+1, interact_targets[agent0])
            #     agent0_in_agent1 = jnp.isin(agent0+1, interact_targets[agent1])
            #     pickup0 = state.agent_inventories[agent0].sum() > INTERACT_THRESHOLD
            #     pickup1 = state.agent_inventories[agent1].sum() > INTERACT_THRESHOLD

            #     print('##')
            #     print(agent0, agent1, 'agent0, agent1:')
            #     print(agent1_in_agent0, 'agent1_in_agent0')
            #     print(agent0_in_agent1, 'agent0_in_agent1')
            #     print(pickup0, 'pickup0')
            #     print(pickup1, 'pickup1')
            #     print('##')



            agent_interaction_mask, _, _, _,pairwise_mask, _ = jax.lax.fori_loop(0, num_agents, process_agent_interactions, 
                                                          (agent_interaction_mask,
                                                           shuffled_indices,
                                                           interact_targets,
                                                           state.agent_inventories, pairwise_mask, zaps))


            # def get_interaction_mask(pair, agent_interaction_mask):
            #     i, j = pair[0], pair[1]
            #     return agent_interaction_mask[i] & agent_interaction_mask[j]

            # interaction_masks = jax.vmap(get_interaction_mask, in_axes=(0, None))(agent_pairs, agent_interaction_mask)
            interaction_masks = pairwise_mask
            # print(agent_interaction_mask, 'agent interaction mask')
            # print(interaction_masks, 'interaction masks')
            # print(interaction_masks.shape, 'interaction mask shape')
            # selected_interactions = interaction_masks.all(axis=-1)
            # print(selected_interactions.shape, 'selected_interactions shape')

            def compute_rewards_for_pair(state, pair):
                return _get_reward(state, pair)  # Assume _get_reward is defined elsewhere

            rewards_for_pairs = jax.vmap(compute_rewards_for_pair, in_axes=(None, 0))(state, agent_pairs)

            def get_selected_pairs(interaction_masks,  agent_pairs, rewards_for_pairs):
                selected_pairs = jnp.where(interaction_masks==1, agent_pairs, -1*jnp.ones((2,), dtype=jnp.int8))
                selected_rewards = jnp.where(interaction_masks==1, rewards_for_pairs, 0)
                return  selected_pairs, selected_rewards
            
            
            vmap_get_selected_pairs = jax.vmap(get_selected_pairs, (0,0,0), (0,0))
            selected_pairs, selected_rewards = vmap_get_selected_pairs(interaction_masks, agent_pairs, rewards_for_pairs)
            # print(selected_pairs.shape, 'selected pairs shape')

            rewards = jnp.zeros((num_agents,))
            # print(rewards_for_pairs.shape,'rewards for pairs')
            def assign_reward(i, val):
                rewards, agent_pairs, selected_rewards = val
                agent0, agent1 = agent_pairs[i]
                rewards = rewards.at[agent0].set(selected_rewards[i][0])
                rewards = rewards.at[agent1].set(selected_rewards[i][1])
                return rewards, agent_pairs, selected_rewards
            
            rewards, _, _ = jax.lax.fori_loop(0, selected_rewards.shape[0], assign_reward, (rewards, agent_pairs, selected_rewards))
            # rewards = rewards.at[selected_pairs.ravel()].set(rewards_for_pairs.ravel())
            # print(rewards.shape, 'rewards shape')
            # Update the grid with the selected interact positions
            # def update_grid(i, val):
            #     grid, selected_interact_positions, interact_idx = val
            #     grid = grid.at[selected_interact_positions[i, 0, 0], selected_interact_positions[i, 0, 1]].set(interact_idx)
            #     grid = grid.at[selected_interact_positions[i, 1, 0], selected_interact_positions[i, 1, 1]].set(interact_idx)
            #     return grid, selected_interact_positions, interact_idx
            
            # grid, _, _ = jax.lax.fori_loop(0, selected_interact_positions.shape[0], update_grid, (state.grid, selected_interact_positions, interact_idx))

            # Update state
            state = state.replace(grid=grid)
            def update_interacted_agents(i, val):
                interacted_agents, pairs = val
                # IDFK IF THIS WORKS. DOES IT AUTOMATICALLY TURN IT INTO BOOL???
                interacted_agents = interacted_agents.at[pairs[i, 0], pairs[i, 1]].set(selected_pairs[i,0])
                interacted_agents = interacted_agents.at[pairs[i, 1], pairs[i, 0]].set(selected_pairs[i,0])
                return interacted_agents, pairs

            # Initialize an interacted_agents matrix with False values
            interacted_agents = jnp.zeros((num_agents, num_agents), dtype=bool)

            # Update interacted_agents matrix using lax.scan and selected_pairs
            interacted_agents, _ = jax.lax.fori_loop(0, selected_pairs.shape[0], update_interacted_agents, (interacted_agents, selected_pairs))
            # print(interacted_agents, 'interacted agents')

            return rewards, state, agent_interaction_mask


        def _step(
            key: chex.PRNGKey,
            state: State,
            actions: Tuple[int, int],
        ):

            """Step the environment."""
            actions = jnp.array([actions[i] for i in self.agents])
            original_agent_pos = state.agent_positions
            # print(original_agent_pos, 'original agent pos')
            # freeze check
            # action_0, action_1 = actions
            # action_0 = jnp.where(state.freeze > 0, Actions.stay, action_0)
            # action_1 = jnp.where(state.freeze > 0, Actions.stay, action_1)
            def freeze_check(action, freeze):
                action = jnp.where(freeze > 0, Actions.stay, action)
                return action
            vmap_freeze_check = jax.vmap(freeze_check, (0, 0), 0)
            actions = vmap_freeze_check(jnp.array(actions), state.agent_freezes)
            def turning_and_moving(agent_pos, action):
                # turning red
                new_pos = jnp.int8(
                    (agent_pos + ROTATIONS[action])
                    % jnp.array([GRID_SIZE + 1, GRID_SIZE + 1, 4])
                )

                # get new positions
                # moving red 
                move = action == Actions.forward
                new_pos = jnp.where(
                    move, new_pos + STEP[agent_pos[2]], new_pos
                )
                new_agent_pos = jnp.clip(
                    new_pos,
                    a_min=jnp.array([0, 0, 0], dtype=jnp.int8),
                    a_max=jnp.array(
                        [GRID_SIZE - 1, GRID_SIZE - 1, 3], dtype=jnp.int8
                    ),
                )

                # if you bounced back to ur original space, we change your move to stay (for collision logic)
                move = (new_agent_pos[:2] != agent_pos[:2]).any()
                return new_agent_pos, move

            vmap_turning_and_moving = jax.vmap(turning_and_moving, (0, 0), (0, 0))
            new_agent_positions, moves = vmap_turning_and_moving(state.agent_positions, actions)


            def check_collision(agent1_pos, agent2_pos):
                collision = jnp.all(agent1_pos[:2] == agent2_pos[:2])
                return collision

            #check if there are collisions
            vmap_check_collision = jax.vmap(jax.vmap(check_collision, (None, 0), 0), (0, None), 0)
            collisions = vmap_check_collision(new_agent_positions, new_agent_positions)

            # if collision, priority to whoever didn't move
            # if collision and we are trying to move to a taken spot we stay
            def update_position(new_agent_pos, old_agent_pos, move1, move2, collision):
                new_agent_pos = jnp.where(
                collision
                * move1
                * (1 - move2),  # red moved, blue didn't
                old_agent_pos,
                new_agent_pos,
                )
                return new_agent_pos

            n = len(moves)
            mask = jnp.eye(n, dtype=bool)
            row_selector = ~mask
            move_matrix = jnp.where(row_selector, moves, moves[0])[:, 1:]

            n = collisions.shape[0]
            mask = jnp.eye(n, dtype=bool)
            row_selector = ~mask
            collision_matrix = jnp.where(row_selector, collisions, collisions[0])[:, 1:]
            # print(collision_matrix, 'collision matrix')
            # print(new_agent_positions, 'new agent positions')
            # print(state.agent_positions, 'old agent positions')

            # print('moves', moves)
            # print('move matrix', move_matrix)
            # print('collisions', collisions)
            # print('collision matrix', collision_matrix)
            vmap_update_position = jax.vmap(jax.vmap(update_position, (None, None, None, 0, 0), 0), (0, 0, 0, 0, 0), 0)
            updated_position = vmap_update_position(new_agent_positions, state.agent_positions, moves, move_matrix, collision_matrix)

            def process_positions(positions, old_pos, new_agent_position):
                # Check if all positions along axis 1 are the same.
                same_positions = jnp.all(jnp.all(positions[1:] == positions[:-1], axis=0), axis=-1)

                # Compute the result by selecting either the first position or the predefined value.
                result = jnp.where(same_positions, new_agent_position, old_pos)
                return result
            
            vmap_process_positions = jax.vmap(process_positions, (0, 0, 0), 0)
            updated_position_coll = vmap_process_positions(updated_position, state.agent_positions, new_agent_positions)

            key, subkey = jax.random.split(key)
            choices = jnp.array([0, 1])
            upper_triangle_indices = jnp.triu_indices(n, k=1)
            upper_triangle_values = jax.random.choice(subkey, choices, shape=(n * (n - 1) // 2,))
            lower_triangle_values = (upper_triangle_values + 1 )% 2
            
            upper_triangle_matrix = jnp.zeros((n, n))
            upper_triangle_matrix = upper_triangle_matrix.at[upper_triangle_indices].set(upper_triangle_values)
            lower_triangle_matrix = jnp.zeros((n, n))
            lower_triangle_matrix = lower_triangle_matrix.at[upper_triangle_indices].set(lower_triangle_values) #reusing upper indices and then transposing
            # utm_transposed = upper_triangle_matrix.T + 1 % 2
            
            takes_square_matrix = upper_triangle_matrix + lower_triangle_matrix.T
            # print(takes_square_matrix, 'takes square matrix')
            # print(takes_square_matrix, 'takes square matrix 1')

            n = takes_square_matrix.shape[0]
            mask = jnp.eye(n, dtype=bool)
            row_selector = ~mask
            # whatever = takes_square_matrix[~jnp.eye(takes_square_matrix.shape[0], dtype=bool)].reshape(takes_square_matrix.shape[0], takes_square_matrix.shape[0]-1)
            takes_square_matrix = jnp.where(row_selector, takes_square_matrix, takes_square_matrix[0])[:, 1:]
            # alex1 = takes_square_matrix.reshape(1,takes_square_matrix.shape[0]* takes_square_matrix.shape[0])
            # alex2 = jnp.delete(alex1, jnp.arange(takes_square_matrix.shape[0])*takes_square_matrix.shape[0])
            # alex3 = alex2.reshape(takes_square_matrix.shape[0], takes_square_matrix.shape[0]-1)
            # takes_square_matrix = jnp.where(row_selector, takes_square_matrix, takes_square_matrix[0])
            # upper = takes_square_matrix[jnp.triu_indices(n, k=1)]
            # upper = takes_square_matrix[:, 1:]
            # lower = takes_square_matrix[1:, :takes_square_matrix.shape[0]-1]
            # takes_square_matrix = jnp.concatenate((upper, lower), axis=1)
            # jax.debug.breakpoint()

            # whatever = jnp.delete(takes_square_matrix, jnp.arange(3), axis=0)
            
            # print(takes_square_matrix, 'takes square matrix 2')
            def update_rand_pos(pos_old, pos_new, collision, move1, move2, takes_square):
                new_pos = jnp.where(
                collision
                * move1
                * move2
                * (
                    1 - takes_square
                ),  # if both collide and red doesn't take square
                pos_old,
                pos_new,
                )
                return new_pos
            # print(takes_square_matrix, 'takes square matrix 2')
            vmap_update_rand_pos = jax.vmap(jax.vmap(update_rand_pos, (None, None, 0, None, 0, 0), 0), (0, 0, 0, 0, 0, 0), 0)
            updated_position = vmap_update_rand_pos(state.agent_positions, updated_position_coll, collision_matrix, moves, move_matrix, takes_square_matrix)
            updated_position = vmap_process_positions(updated_position, state.agent_positions, updated_position_coll)
            def update_inventories(new_pos, inventory):
                coop_matches = (
                state.grid[new_pos[0], new_pos[1]] == Items.cooperation_coin
                )
                defect_matches = (
                state.grid[new_pos[0], new_pos[1]] == Items.defection_coin
                )
                inventory = inventory + jnp.array(
                [coop_matches, defect_matches]
                )
                return inventory

            vmap_update_inventory = jax.vmap(update_inventories, (0, 0), 0)

            agent_inventories = vmap_update_inventory(updated_position, state.agent_inventories)

            state = state.replace(agent_inventories=agent_inventories)
            state = state.replace(agent_positions=updated_position)

            key, subkey = jax.random.split(key)
            red_reward, blue_reward = 0, 0
            (
                rewards, 
                state, 
                interacted_agents
            ) = _interact(state, actions, key)
            # print(interacted_agents, 'interacted agents')

            def update_freeze(freeze, interact):
                freeze = jnp.where(
                interact, freeze_penalty, freeze
                )
                return freeze
            vmap_update_freeze = jax.vmap(update_freeze, (0, 0), 0)

            # interact_vector = interacted_agents.any(axis=1)
            state = state.replace(agent_freezes=state.agent_freezes-1)

            agent_freezes = vmap_update_freeze(state.agent_freezes, interacted_agents)
            state = state.replace(agent_freezes=agent_freezes)

            # state_sft_re = _soft_reset_state(key, state)
            key, subkey = jax.random.split(key)

            agent_pos = jax.random.choice(
                subkey, AGENT_SPAWNS, shape=(), replace=False
            )
            player_dir = jax.random.randint(
                subkey, shape=(num_agents,), minval=0, maxval=3, dtype=jnp.int8
            )
            sft_re_player_pos = jnp.array(
                [agent_pos[:num_agents, 0], agent_pos[:num_agents, 1], player_dir]
            ).T

            # print(sft_re_player_pos, 'sft_re_player_pos')
            def update_positions_freezes(new_player_pos, agent_positions, agent_freezes, 
                                        agent_inventories,):
                agent_positions = jnp.where(
                                        agent_freezes==0,
                                        new_player_pos,
                                        agent_positions   
                                    )
                agent_inventories = jnp.where(
                                        agent_freezes==0,
                                        jnp.zeros((2,), dtype=jnp.int8),
                                        agent_inventories)
                agent_freezes = jnp.where(
                                        agent_freezes==0,
                                        -1,
                                        agent_freezes   
                                    )
                return agent_positions, agent_freezes, agent_inventories
            
            vmap_update_positions_freezes = jax.vmap(update_positions_freezes, (0, 0, 0, 0), (0,0,0))
            agent_positions, agent_freezes, agent_inventories = vmap_update_positions_freezes(sft_re_player_pos, 
                                                                state.agent_positions, state.agent_freezes, state.agent_inventories)
            # print(agent_freezes, 'agent freezes')
            # print(agent_inventories, 'agent inventories')
            # print(agent_positions, 'agent positions')

            # print(agent_positions, 'agent positions after freezes')

            def pos_remove(i, val):
                grid, old_pos = val
                grid = grid.at[
                    (old_pos[i, 0], old_pos[i,1])
                ].set(jnp.int8(Items.empty))
                return (grid, old_pos)

            def pos_add(i, val):
                grid, new_pos = val
                grid = grid.at[(new_pos[i,0], new_pos[i,1])].set(
                    jnp.int8(i+1)
                )
                return (grid, new_pos)

            grid, _ = jax.lax.fori_loop(0, num_agents, pos_remove, (state.grid, original_agent_pos))
            grid, _ = jax.lax.fori_loop(0, num_agents, pos_add, (grid, agent_positions))

            state = state.replace(coin_timer_coop=state.coin_timer_coop-1)
            state = state.replace(coin_timer_defect=state.coin_timer_defect-1)

            # define jax foriloop function to update grid cell to coin value when cointimer hits 0 but only if that cell is 0, otherwise take cell value
            def update_grid_coin_coop(i, val):
                grid, coin_timer_coop, coop_coin = val
                cond = jnp.logical_and(coin_timer_coop[i] == 0, grid[coop_coin[i, 0], coop_coin[i, 1]] == 0)
                update_value = jnp.where(cond, Items.cooperation_coin, grid[coop_coin[i, 0], coop_coin[i, 1]])
                updated_grid = grid.at[coop_coin[i, 0], coop_coin[i, 1]].set(update_value)
                return updated_grid, coin_timer_coop, coop_coin

            def update_grid_coin_defect(i, val):
                grid, coin_timer_defect, defect_coin = val
                cond = jnp.logical_and(coin_timer_defect[i] == 0, grid[defect_coin[i, 0], defect_coin[i, 1]] == 0)
                update_value = jnp.where(cond, Items.defection_coin, grid[defect_coin[i, 0], defect_coin[i, 1]])
                updated_grid = grid.at[defect_coin[i, 0], defect_coin[i, 1]].set(update_value)
                return updated_grid, coin_timer_defect, defect_coin

            # now write the foriloop for the above functions
            grid, _, _ = jax.lax.fori_loop(0, NUM_COINS, update_grid_coin_coop, (grid, state.coin_timer_coop, state.coin_coop))
            grid, _, _ = jax.lax.fori_loop(0, NUM_COINS, update_grid_coin_defect, (grid, state.coin_timer_defect, state.coin_defect))


            key, subkey = jax.random.split(key)
            new_coop_coin_timer = jax.random.randint(
                subkey, shape=(NUM_COINS,), minval=3, maxval=10, dtype=jnp.int8
            )
            key, subkey = jax.random.split(key)
            new_defect_coin_timer = jax.random.randint(
                subkey, shape=(NUM_COINS,), minval=3, maxval=10, dtype=jnp.int8
            )

            def update_timers(coop_coin_timer, defect_coin_timer, new_coop_coin_timer, new_defect_coin_timer):
                coop_coin_timer = jnp.where(coop_coin_timer==0, new_coop_coin_timer, coop_coin_timer)
                defect_coin_timer = jnp.where(defect_coin_timer==0, new_defect_coin_timer, defect_coin_timer)
                return coop_coin_timer, defect_coin_timer

            vmap_update_timers = jax.vmap(update_timers, (0, 0, 0, 0), (0, 0))
            coin_timer_coop, coin_timer_defect = vmap_update_timers(state.coin_timer_coop, state.coin_timer_defect, new_coop_coin_timer, new_defect_coin_timer)

            state = state.replace(agent_freezes=agent_freezes)
            state = state.replace(agent_positions=agent_positions)
            state = state.replace(grid=grid)
            state = state.replace(agent_inventories=agent_inventories)
            state = state.replace(coin_timer_coop=coin_timer_coop)
            state = state.replace(coin_timer_defect=coin_timer_defect)

            state_nxt = State(
                agent_positions=state.agent_positions,
                inner_t=state.inner_t + 1,
                outer_t=state.outer_t,
                grid=state.grid,
                agent_inventories=state.agent_inventories,
                coin_coop=state.coin_coop,
                coin_defect=state.coin_defect,
                agent_freezes=state.agent_freezes,
                coin_timer_coop=state.coin_timer_coop,
                coin_timer_defect=state.coin_timer_defect,
            )

            # now calculate if done for inner or outer episode
            inner_t = state_nxt.inner_t
            outer_t = state_nxt.outer_t
            reset_inner = inner_t == num_inner_steps

            # # # if inner episode is done, return start state for next game
            state_re = _reset_state(key)
            state_re = state_re.replace(outer_t=outer_t + 1)
            state = jax.tree_map(
                lambda x, y: jax.lax.select(reset_inner, x, y),
                state_re,
                state_nxt,
            )
            outer_t = state.outer_t
            reset_outer = outer_t == num_outer_steps
            done = {}
            done["__all__"] = reset_outer

            obs = _get_obs(state)
            rewards = jnp.where(reset_inner, 0, rewards)

            return (
                obs,
                state,
                rewards,
                done,
                {"discount": jnp.zeros((), dtype=jnp.int8)},
            )

        def _reset_state(
            key: jnp.ndarray,
        ) -> Tuple[jnp.ndarray, State]:
            key, subkey = jax.random.split(key)

            # coin_pos = jax.random.choice(
            #     subkey, COIN_SPAWNS, shape=(NUM_COIN_TYPES*NUM_COINS,), replace=False
            # )

            agent_pos = jax.random.choice(
                subkey, AGENT_SPAWNS, shape=(), replace=False
            )
            player_dir = jax.random.randint(
                subkey, shape=(num_agents,), minval=0, maxval=3, dtype=jnp.int8
            )
            player_pos = jnp.array(
                [agent_pos[:num_agents, 0], agent_pos[:num_agents, 1], player_dir]
            ).T

            grid = jnp.zeros((GRID_SIZE, GRID_SIZE), jnp.int8)

            def pos_fun(i, val):
                grid, player_pos = val
                grid = grid.at[player_pos[i, 0], player_pos[i, 1]].set(
                jnp.int8(i+1))
                return (grid, player_pos)

            grid, _ = jax.lax.fori_loop(0, num_agents, pos_fun, (grid, player_pos))

            if fixed_coin_location:
                rand_idx = jax.random.randint(
                    subkey, shape=(), minval=0, maxval=1
                )
                red_coins = jnp.where(rand_idx, RED_SPAWN, BLUE_SPAWN)
                blue_coins = jnp.where(rand_idx, BLUE_SPAWN, RED_SPAWN)
            else:
                coin_spawn = jax.random.permutation(
                    subkey, COIN_SPAWNS, axis=0
                )
                red_coins = coin_spawn[:NUM_COINS, :]
                blue_coins = coin_spawn[NUM_COINS:, :]

            for i in range(NUM_COINS):
                grid = grid.at[red_coins[i, 0], red_coins[i, 1]].set(
                    jnp.int8(Items.defection_coin)
                )

            for i in range(NUM_COINS):
                grid = grid.at[blue_coins[i, 0], blue_coins[i, 1]].set(
                    jnp.int8(Items.cooperation_coin)
                )
            key, subkey = jax.random.split(key)
            #initialize coin timer to random numbers between 1 and 10
            coin_timer_coop = jax.random.randint(
                subkey, shape=(NUM_COINS,), minval=3, maxval=10
            )
            key, subkey = jax.random.split(key)
            coin_timer_defect = jax.random.randint(
                subkey, shape=(NUM_COINS,), minval=3, maxval=10
            )

            return State(
                agent_positions=player_pos,
                inner_t=0,
                outer_t=0,
                grid=grid,
                agent_inventories=jnp.zeros((num_agents, 2)),
                coin_defect=red_coins,
                coin_coop=blue_coins,
                agent_freezes=jnp.int16(-1*jnp.ones((num_agents,))),
                coin_timer_coop=coin_timer_coop,
                coin_timer_defect=coin_timer_defect,
            )

        def reset(
            key: jnp.ndarray,
        ) -> Tuple[jnp.ndarray, State]:
            state = _reset_state(key)
            obs = _get_obs(state)
            return obs, state

        # overwrite Gymnax as it makes single-agent assumptions
        # self.step = jax.jit(_step)
        self.step_env = _step
        self.reset = jax.jit(reset)
        self.get_obs_point = _get_obs_point
        self.get_reward = _get_reward

        # for debugging
        self.get_obs = _get_obs
        self.cnn = True

        self.num_inner_steps = num_inner_steps
        self.num_outer_steps = num_outer_steps
        self.num_agents = num_agents
        _shape = (
            (OBS_SIZE, OBS_SIZE, len(Items) - 1 + 4)
            if self.cnn
            else (OBS_SIZE**2 * (len(Items) - 1 + 4),)
        )
        self.observation_spaces = {
            i: {"observation": spaces.Box(
                low=0, high=1, shape=_shape, dtype=jnp.uint8),
            "inventory": spaces.Box(
                low=0,high=NUM_COINS,shape=NUM_COIN_TYPES + 4,dtype=jnp.uint8,),
        } for i in range(self.num_agents)}
        self.action_spaces = {
            i: spaces.Discrete(len(Actions)) for i in range(self.num_agents)
        }

    @property
    def name(self) -> str:
        """Environment name."""
        return "MGinTheGrid"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return len(Actions)


    def state_space(self) -> spaces.Dict:
        """State space of the environment."""
        _shape = (
            (GRID_SIZE, GRID_SIZE, NUM_TYPES + 4)
            if self.cnn
            else (GRID_SIZE**2 * (NUM_TYPES + 4),)
        )
        return spaces.Box(low=0, high=1, shape=_shape, dtype=jnp.uint8)

    @classmethod
    def render_tile(
        cls,
        obj: int,
        agent_dir: Union[int, None] = None,
        agent_hat: bool = False,
        highlight: bool = False,
        tile_size: int = 32,
        subdivs: int = 3,
    ) -> onp.ndarray:
        """
        Render a tile and cache the result
        """

        # Hash map lookup key for the cache
        key: tuple[Any, ...] = (agent_dir, agent_hat, highlight, tile_size)
        if obj:
            key = (obj, 0, 0, 0) + key if obj else key

        if key in cls.tile_cache:
            return cls.tile_cache[key]

        img = onp.zeros(
            shape=(tile_size * subdivs, tile_size * subdivs, 3),
            dtype=onp.uint8,
        )

        # Draw the grid lines (top and left edges)
        fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
        fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))
    # class Items(IntEnum):

        if obj == Items.agent1:
            # Draw the agent 1
            agent_color = PLAYER1_COLOUR
        elif obj == Items.agent2:
            # Draw agent 2
            agent_color = PLAYER2_COLOUR
        elif obj == Items.agent3:
            # Draw agent 3
            agent_color = PLAYER3_COLOUR
        elif obj == Items.agent4:
            # Draw agent 4
            agent_color = PLAYER4_COLOUR
        elif obj == Items.agent5:
            # Draw agent 5
            agent_color = PLAYER5_COLOUR
        elif obj == Items.agent6:
            # Draw agent 6
            agent_color = PLAYER6_COLOUR
        elif obj == Items.agent7:
            # Draw agent 7
            agent_color = PLAYER7_COLOUR
        elif obj == Items.agent8:
            # Draw agent 8
            agent_color = PLAYER8_COLOUR
        elif obj == Items.defection_coin:
            # Draw the red coin as GREEN COOPERATE
            fill_coords(
                img, point_in_circle(0.5, 0.5, 0.31), (44.0, 160.0, 44.0)
            )
        elif obj == Items.cooperation_coin:
            # Draw the blue coin as DEFECT/ RED COIN
            fill_coords(
                img, point_in_circle(0.5, 0.5, 0.31), (214.0, 39.0, 40.0)
            )
        elif obj == Items.wall:
            fill_coords(img, point_in_rect(0, 1, 0, 1), (127.0, 127.0, 127.0))

        elif obj == Items.interact:
            fill_coords(img, point_in_rect(0, 1, 0, 1), (188.0, 189.0, 34.0))

        elif obj == 99:
            fill_coords(img, point_in_rect(0, 1, 0, 1), (44.0, 160.0, 44.0))

        elif obj == 100:
            fill_coords(img, point_in_rect(0, 1, 0, 1), (214.0, 39.0, 40.0))

        elif obj == 101:
            # white square
            fill_coords(img, point_in_rect(0, 1, 0, 1), (255.0, 255.0, 255.0))

        # Overlay the agent on top
        if agent_dir is not None:
            if agent_hat:
                tri_fn = point_in_triangle(
                    (0.12, 0.19),
                    (0.87, 0.50),
                    (0.12, 0.81),
                    0.3,
                )

                # Rotate the agent based on its direction
                tri_fn = rotate_fn(
                    tri_fn,
                    cx=0.5,
                    cy=0.5,
                    theta=0.5 * math.pi * (1 - agent_dir),
                )
                fill_coords(img, tri_fn, (255.0, 255.0, 255.0))

            tri_fn = point_in_triangle(
                (0.12, 0.19),
                (0.87, 0.50),
                (0.12, 0.81),
                0.0,
            )

            # Rotate the agent based on its direction
            tri_fn = rotate_fn(
                tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * (1 - agent_dir)
            )
            fill_coords(img, tri_fn, agent_color)

        # Highlight the cell if needed
        if highlight:
            highlight_img(img)

        # Downsample the image to perform supersampling/anti-aliasing
        img = downsample(img, subdivs)

        # Cache the rendered tile
        cls.tile_cache[key] = img
        return img

    def render(
        self,
        state: State,
    ) -> onp.ndarray:
        """
        Render this grid at a given scale
        :param r: target renderer object
        :param tile_size: tile size in pixels
        """
        tile_size = 32
        highlight_mask = onp.zeros_like(onp.array(GRID))

        # Compute the total grid size
        width_px = GRID.shape[0] * tile_size
        height_px = GRID.shape[0] * tile_size

        img = onp.zeros(shape=(height_px, width_px, 3), dtype=onp.uint8)
        grid = onp.array(state.grid)
        grid = onp.pad(
            grid, ((PADDING, PADDING), (PADDING, PADDING)), constant_values=Items.wall
        )
        for a in range(self.num_agents):
            startx, starty = self.get_obs_point(
                state.agent_positions[a, 0], 
                state.agent_positions[a, 1], 
                state.agent_positions[a, 2]
            )
            highlight_mask[
                startx : startx + OBS_SIZE, starty : starty + OBS_SIZE
            ] = True

        # Render the grid
        for j in range(0, grid.shape[1]):
            for i in range(0, grid.shape[0]):
                cell = grid[i, j]
                if cell == 0:
                    cell = None
                agent_here = []
                for a in range(1, self.num_agents+1):
                    agent_here.append(cell == a)
                # if cell in [1,2]:
                #     print(f'coordinates: {i},{j}')
                #     print(cell)


                agent_dir = None
                for a in range(self.num_agents):
                    agent_dir = (
                        state.agent_positions[a,2].item()
                        if agent_here[a]
                        else agent_dir
                    )
                
                agent_hat = False
                for a in range(self.num_agents):
                    agent_hat = (
                        bool(state.agent_inventories[a].sum() > INTERACT_THRESHOLD)
                        if agent_here[a]
                        else agent_hat
                    )

                tile_img = InTheGrid.render_tile(
                    cell,
                    agent_dir=agent_dir,
                    agent_hat=agent_hat,
                    highlight=highlight_mask[i, j],
                    tile_size=tile_size,
                )

                ymin = j * tile_size
                ymax = (j + 1) * tile_size
                xmin = i * tile_size
                xmax = (i + 1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img

        img = onp.rot90(
            img[
                (PADDING - 1) * tile_size : -(PADDING - 1) * tile_size,
                (PADDING - 1) * tile_size : -(PADDING - 1) * tile_size,
                :,
            ],
            2,
        )
        # Render the inventory
        # agent_inv = []
        # for a in range(self.num_agents):
        #     agent_inv.append(self.render_inventory(state.agent_inventories[a], img.shape[1]))


        time = self.render_time(state, img.shape[1])
        # img = onp.concatenate((img, *agent_inv, time), axis=0)
        img = onp.concatenate((img, time), axis=0)
        return img

    def render_inventory(self, inventory, width_px) -> onp.array:
        tile_height = 32
        height_px = NUM_COIN_TYPES * tile_height
        img = onp.zeros(shape=(height_px, width_px, 3), dtype=onp.uint8)
        tile_width = width_px // NUM_COINS
        for j in range(0, NUM_COIN_TYPES):
            num_coins = inventory[j]
            for i in range(int(num_coins)):
                cell = None
                if j == 0:
                    cell = 99
                elif j == 1:
                    cell = 100
                tile_img = InTheGrid.render_tile(cell, tile_size=tile_height)
                ymin = j * tile_height
                ymax = (j + 1) * tile_height
                xmin = i * tile_width
                xmax = (i + 1) * tile_width
                img[ymin:ymax, xmin:xmax, :] = onp.resize(
                    tile_img, (tile_height, tile_width, 3)
                )
        return img

    def render_time(self, state, width_px) -> onp.array:
        inner_t = state.inner_t
        outer_t = state.outer_t
        tile_height = 32
        img = onp.zeros(shape=(2 * tile_height, width_px, 3), dtype=onp.uint8)
        tile_width = width_px // (self.num_inner_steps)
        j = 0
        for i in range(0, inner_t):
            ymin = j * tile_height
            ymax = (j + 1) * tile_height
            xmin = i * tile_width
            xmax = (i + 1) * tile_width
            img[ymin:ymax, xmin:xmax, :] = onp.int8(255)
        tile_width = width_px // (self.num_outer_steps)
        j = 1
        for i in range(0, outer_t):
            ymin = j * tile_height
            ymax = (j + 1) * tile_height
            xmin = i * tile_width
            xmax = (i + 1) * tile_width
            img[ymin:ymax, xmin:xmax, :] = onp.int8(255)
        return img


