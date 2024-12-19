from enum import IntEnum
import math
from typing import Any, Optional, Tuple, Union, Dict

import chex
import jax
import jax.numpy as jnp
import numpy as onp
from flax.struct import dataclass

from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.environments import spaces


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
    red_pos: jnp.ndarray
    blue_pos: jnp.ndarray
    inner_t: int
    outer_t: int
    grid: jnp.ndarray
    red_inventory: jnp.ndarray
    blue_inventory: jnp.ndarray
    red_coins: jnp.ndarray
    blue_coins: jnp.ndarray
    freeze: int


@chex.dataclass
class EnvParams:
    payoff_matrix: chex.ArrayDevice
    freeze_penalty: int


class Actions(IntEnum):
    left = 0
    right = 1
    forward = 2
    interact = 3
    stay = 4


class Items(IntEnum):
    empty = 0
    red_agent = 1
    blue_agent = 2
    red_coin = 3
    blue_coin = 4
    wall = 5
    interact = 6


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

AGENT_SPAWNS = jnp.array(
    [
        [(j, i), (GRID_SIZE - 1 - j, GRID_SIZE - 1 - i)]
        for (i, j) in AGENT_SPAWNS
    ],
    dtype=jnp.int8,
).reshape(-1, 2, 2)


PLAYER1_COLOUR = (255.0, 127.0, 14.0)
PLAYER2_COLOUR = (31.0, 119.0, 180.0)
GREEN_COLOUR = (44.0, 160.0, 44.0)
RED_COLOUR = (214.0, 39.0, 40.0)


class InTheGrid_2p(MultiAgentEnv):
    """
    JAX Compatible version of *inTheMatix environment.
    """

    # used for caching
    tile_cache: Dict[Tuple[Any, ...], Any] = {}

    def __init__(
        self,
        num_inner_steps=152,
        num_outer_steps=1,
        fixed_coin_location=True,
        num_agents=2,
        payoff_matrix=jnp.array([[[3, 0], [5, 1]], [[3, 5], [0, 1]]]),
        freeze_penalty=5
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
            x, y = _get_obs_point(
                state.red_pos[0], state.red_pos[1], state.red_pos[2]
            )
            grid1 = jax.lax.dynamic_slice(
                grid,
                start_indices=(x, y),
                slice_sizes=(OBS_SIZE, OBS_SIZE),
            )
            # rotate
            grid1 = jnp.where(
                state.red_pos[2] == 1,
                jnp.rot90(grid1, k=1, axes=(0, 1)),
                grid1,
            )
            grid1 = jnp.where(
                state.red_pos[2] == 2,
                jnp.rot90(grid1, k=2, axes=(0, 1)),
                grid1,
            )
            grid1 = jnp.where(
                state.red_pos[2] == 3,
                jnp.rot90(grid1, k=3, axes=(0, 1)),
                grid1,
            )

            angle1 = -1 * jnp.ones_like(grid1, dtype=jnp.int8)
            angle1 = jnp.where(
                grid1 == Items.blue_agent,
                (state.blue_pos[2] - state.red_pos[2]) % 4,
                -1,
            )
            angle1 = jax.nn.one_hot(angle1, 4)

            # one-hot (drop first channel as its empty blocks)
            grid1 = jax.nn.one_hot(grid1 - 1, len(Items) - 1, dtype=jnp.int8)
            obs1 = jnp.concatenate([grid1, angle1], axis=-1)

            x, y = _get_obs_point(
                state.blue_pos[0], state.blue_pos[1], state.blue_pos[2]
            )

            grid2 = jax.lax.dynamic_slice(
                grid,
                start_indices=(x, y),
                slice_sizes=(OBS_SIZE, OBS_SIZE),
            )

            grid2 = jnp.where(
                state.blue_pos[2] == 1,
                jnp.rot90(grid2, k=1, axes=(0, 1)),
                grid2,
            )
            grid2 = jnp.where(
                state.blue_pos[2] == 2,
                jnp.rot90(grid2, k=2, axes=(0, 1)),
                grid2,
            )
            grid2 = jnp.where(
                state.blue_pos[2] == 3,
                jnp.rot90(grid2, k=3, axes=(0, 1)),
                grid2,
            )

            angle2 = -1 * jnp.ones_like(grid2, dtype=jnp.int8)
            angle2 = jnp.where(
                grid2 == Items.red_agent,
                (state.red_pos[2] - state.blue_pos[2]) % 4,
                -1,
            )
            angle2 = jax.nn.one_hot(angle2, 4)

            # sends 0 -> -1 and droped by one_hot
            grid2 = jax.nn.one_hot(grid2 - 1, len(Items) - 1, dtype=jnp.int8)
            # make agent 2 think it is agent 1
            _grid2 = grid2.at[:, :, 0].set(grid2[:, :, 1])
            _grid2 = _grid2.at[:, :, 1].set(grid2[:, :, 0])
            _obs2 = jnp.concatenate([_grid2, angle2], axis=-1)

            red_pickup = jnp.sum(state.red_inventory) > INTERACT_THRESHOLD
            blue_pickup = jnp.sum(state.blue_inventory) > INTERACT_THRESHOLD

            blue_to_show = jnp.where(
                state.freeze >= 0, state.blue_inventory, 0
            )
            red_to_show = jnp.where(state.freeze >= 0, state.red_inventory, 0)

            return {
                "observation": obs1,
                "inventory": jnp.array(
                    [
                        state.red_inventory[0],
                        state.red_inventory[1],
                        red_pickup,
                        blue_pickup,
                        blue_to_show[0],
                        blue_to_show[1],
                    ],
                    dtype=jnp.int8,
                ),
            }, {
                "observation": _obs2,
                "inventory": jnp.array(
                    [
                        state.blue_inventory[0],
                        state.blue_inventory[1],
                        blue_pickup,
                        red_pickup,
                        red_to_show[0],
                        red_to_show[1],
                    ],
                    dtype=jnp.int8,
                ),
            }

        def _get_reward(state: State) -> jnp.ndarray:
            inv1 = state.red_inventory / state.red_inventory.sum()
            inv2 = state.blue_inventory / state.blue_inventory.sum()
            r1 = inv1 @ payoff_matrix[0] @ inv2.T
            r2 = inv1 @ payoff_matrix[1] @ inv2.T
            return r1, r2

        def _interact(
            state: State, actions: Tuple[int, int]
        ) -> Tuple[bool, float, float, State]:
            # if interact
            a0, a1 = actions

            red_zap = a0 == Actions.interact
            blue_zap = a1 == Actions.interact
            interact_idx = jnp.int8(Items.interact)

            # remove old interacts
            state = state.replace(grid=jnp.where(
                state.grid == interact_idx, jnp.int8(Items.empty), state.grid
            ))

            # check 1 ahead
            red_target = jnp.clip(
                state.red_pos + STEP[state.red_pos[2]], 0, GRID_SIZE - 1
            )
            blue_target = jnp.clip(
                state.blue_pos + STEP[state.blue_pos[2]], 0, GRID_SIZE - 1
            )

            red_interact = (
                state.grid[red_target[0], red_target[1]] == Items.blue_agent
            )
            blue_interact = (
                state.grid[blue_target[0], blue_target[1]] == Items.red_agent
            )

            # check 2 ahead
            red_target_ahead = jnp.clip(
                state.red_pos + 2 * STEP[state.red_pos[2]], 0, GRID_SIZE - 1
            )
            blue_target_ahead = jnp.clip(
                state.blue_pos + 2 * STEP[state.blue_pos[2]], 0, GRID_SIZE - 1
            )

            red_interact_ahead = (
                state.grid[red_target_ahead[0], red_target_ahead[1]]
                == Items.blue_agent
            )
            blue_interact_ahead = (
                state.grid[blue_target_ahead[0], blue_target_ahead[1]]
                == Items.red_agent
            )

            # check to your right  - clip can't be used here as it will wrap down
            red_target_right = (
                state.red_pos
                + STEP[state.red_pos[2]]
                + STEP[(state.red_pos[2] + 1) % 4]
            )
            oob_red = jnp.logical_or(
                (red_target_right > GRID_SIZE - 1).any(),
                (red_target_right < 0).any(),
            )
            red_target_right = jnp.where(oob_red, red_target, red_target_right)

            blue_target_right = (
                state.blue_pos
                + STEP[state.blue_pos[2]]
                + STEP[(state.blue_pos[2] + 1) % 4]
            )
            oob_blue = jnp.logical_or(
                (blue_target_right > GRID_SIZE - 1).any(),
                (blue_target_right < 0).any(),
            )
            blue_target_right = jnp.where(
                oob_blue, blue_target, blue_target_right
            )

            red_interact_right = (
                state.grid[red_target_right[0], red_target_right[1]]
                == Items.blue_agent
            )
            blue_interact_right = (
                state.grid[blue_target_right[0], blue_target_right[1]]
                == Items.red_agent
            )

            # check to your left
            red_target_left = (
                state.red_pos
                + STEP[state.red_pos[2]]
                + STEP[(state.red_pos[2] - 1) % 4]
            )
            oob_red = jnp.logical_or(
                (red_target_left > GRID_SIZE - 1).any(),
                (red_target_left < 0).any(),
            )
            red_target_left = jnp.where(oob_red, red_target, red_target_left)

            blue_target_left = (
                state.blue_pos
                + STEP[state.blue_pos[2]]
                + STEP[(state.blue_pos[2] - 1) % 4]
            )
            oob_blue = jnp.logical_or(
                (blue_target_left > GRID_SIZE - 1).any(),
                (blue_target_left < 0).any(),
            )
            blue_target_left = jnp.where(
                oob_blue, blue_target, blue_target_left
            )

            red_interact_left = (
                state.grid[red_target_left[0], red_target_left[1]]
                == Items.blue_agent
            )
            blue_interact_left = (
                state.grid[blue_target_left[0], blue_target_left[1]]
                == Items.red_agent
            )

            red_interact = jnp.logical_or(
                red_interact,
                jnp.logical_or(
                    red_interact_ahead,
                    jnp.logical_or(red_interact_right, red_interact_left),
                ),
            )

            # update grid with red zaps
            aux_grid = jnp.copy(state.grid)

            item = jnp.where(
                state.grid[red_target[0], red_target[1]],
                state.grid[red_target[0], red_target[1]],
                interact_idx,
            )
            aux_grid = aux_grid.at[red_target[0], red_target[1]].set(item)

            item = jnp.where(
                state.grid[red_target_ahead[0], red_target_ahead[1]],
                state.grid[red_target_ahead[0], red_target_ahead[1]],
                interact_idx,
            )
            aux_grid = aux_grid.at[
                red_target_ahead[0], red_target_ahead[1]
            ].set(item)

            item = jnp.where(
                state.grid[red_target_right[0], red_target_right[1]],
                state.grid[red_target_right[0], red_target_right[1]],
                interact_idx,
            )
            aux_grid = aux_grid.at[
                red_target_right[0], red_target_right[1]
            ].set(item)

            item = jnp.where(
                state.grid[red_target_left[0], red_target_left[1]],
                state.grid[red_target_left[0], red_target_left[1]],
                interact_idx,
            )
            aux_grid = aux_grid.at[red_target_left[0], red_target_left[1]].set(
                item
            )

            state = state.replace(grid=jnp.where(red_zap, aux_grid, state.grid))

            # update grid with blue zaps
            aux_grid = jnp.copy(state.grid)
            blue_interact = jnp.logical_or(
                blue_interact,
                jnp.logical_or(
                    blue_interact_ahead,
                    jnp.logical_or(blue_interact_right, blue_interact_left),
                ),
            )
            item = jnp.where(
                state.grid[blue_target[0], blue_target[1]],
                state.grid[blue_target[0], blue_target[1]],
                interact_idx,
            )
            aux_grid = aux_grid.at[blue_target[0], blue_target[1]].set(item)

            item = jnp.where(
                state.grid[blue_target_ahead[0], blue_target_ahead[1]],
                state.grid[blue_target_ahead[0], blue_target_ahead[1]],
                interact_idx,
            )
            aux_grid = aux_grid.at[
                blue_target_ahead[0], blue_target_ahead[1]
            ].set(item)

            item = jnp.where(
                state.grid[blue_target_right[0], blue_target_right[1]],
                state.grid[blue_target_right[0], blue_target_right[1]],
                interact_idx,
            )
            aux_grid = aux_grid.at[
                blue_target_right[0], blue_target_right[1]
            ].set(item)

            item = jnp.where(
                state.grid[blue_target_left[0], blue_target_left[1]],
                state.grid[blue_target_left[0], blue_target_left[1]],
                interact_idx,
            )
            aux_grid = aux_grid.at[
                blue_target_left[0], blue_target_left[1]
            ].set(item)
            state = state.replace(grid=jnp.where(blue_zap, aux_grid, state.grid))

            # rewards
            red_reward, blue_reward = 0.0, 0.0
            _r_reward, _b_reward = _get_reward(state)

            interact = jnp.logical_or(
                red_zap * red_interact, blue_zap * blue_interact
            )

            red_pickup = state.red_inventory.sum() > INTERACT_THRESHOLD
            blue_pickup = state.blue_inventory.sum() > INTERACT_THRESHOLD
            interact = jnp.logical_and(
                interact, jnp.logical_and(red_pickup, blue_pickup)
            )

            red_reward = jnp.where(
                interact, red_reward + _r_reward, red_reward
            )
            blue_reward = jnp.where(
                interact, blue_reward + _b_reward, blue_reward
            )
            return interact, red_reward, blue_reward, state

        def _step(
            key: chex.PRNGKey,
            state: State,
            actions: Tuple[int, int]
        ):

            """Step the environment."""

            # freeze check
            action_0, action_1 = actions
            action_0 = jnp.where(state.freeze > 0, Actions.stay, action_0)
            action_1 = jnp.where(state.freeze > 0, Actions.stay, action_1)

            # turning red
            new_red_pos = jnp.int8(
                (state.red_pos + ROTATIONS[action_0])
                % jnp.array([GRID_SIZE + 1, GRID_SIZE + 1, 4])
            )

            # moving red
            red_move = action_0 == Actions.forward
            new_red_pos = jnp.where(
                red_move, new_red_pos + STEP[state.red_pos[2]], new_red_pos
            )
            new_red_pos = jnp.clip(
                new_red_pos,
                a_min=jnp.array([0, 0, 0], dtype=jnp.int8),
                a_max=jnp.array(
                    [GRID_SIZE - 1, GRID_SIZE - 1, 3], dtype=jnp.int8
                ),
            )

            # if you bounced back to ur original space, we change your move to stay (for collision logic)
            red_move = (new_red_pos[:2] != state.red_pos[:2]).any()

            # turning blue
            new_blue_pos = jnp.int8(
                (state.blue_pos + ROTATIONS[action_1])
                % jnp.array([GRID_SIZE + 1, GRID_SIZE + 1, 4], dtype=jnp.int8)
            )

            # moving blue
            blue_move = action_1 == Actions.forward
            new_blue_pos = jnp.where(
                blue_move, new_blue_pos + STEP[state.blue_pos[2]], new_blue_pos
            )
            new_blue_pos = jnp.clip(
                new_blue_pos,
                a_min=jnp.array([0, 0, 0], dtype=jnp.int8),
                a_max=jnp.array(
                    [GRID_SIZE - 1, GRID_SIZE - 1, 3], dtype=jnp.int8
                ),
            )
            blue_move = (new_blue_pos[:2] != state.blue_pos[:2]).any()

            # if collision, priority to whoever didn't move
            collision = jnp.all(new_red_pos[:2] == new_blue_pos[:2])

            new_red_pos = jnp.where(
                collision
                * red_move
                * (1 - blue_move),  # red moved, blue didn't
                state.red_pos,
                new_red_pos,
            )
            new_blue_pos = jnp.where(
                collision
                * (1 - red_move)
                * blue_move,  # blue moved, red didn't
                state.blue_pos,
                new_blue_pos,
            )

            # if both moved, then randomise
            red_takes_square = jax.random.choice(key, jnp.array([0, 1]))
            new_red_pos = jnp.where(
                collision
                * blue_move
                * red_move
                * (
                    1 - red_takes_square
                ),  # if both collide and red doesn't take square
                state.red_pos,
                new_red_pos,
            )
            new_blue_pos = jnp.where(
                collision
                * blue_move
                * red_move
                * (
                    red_takes_square
                ),  # if both collide and blue doesn't take square
                state.blue_pos,
                new_blue_pos,
            )

            # update inventories
            red_red_matches = (
                state.grid[new_red_pos[0], new_red_pos[1]] == Items.red_coin
            )
            red_blue_matches = (
                state.grid[new_red_pos[0], new_red_pos[1]] == Items.blue_coin
            )
            blue_red_matches = (
                state.grid[new_blue_pos[0], new_blue_pos[1]] == Items.red_coin
            )
            blue_blue_matches = (
                state.grid[new_blue_pos[0], new_blue_pos[1]] == Items.blue_coin
            )

            state = state.replace(red_inventory=state.red_inventory + jnp.array(
                [red_red_matches, red_blue_matches]
            ))
            state = state.replace(blue_inventory=state.blue_inventory + jnp.array(
                [blue_red_matches, blue_blue_matches]
            ))

            # update grid
            state = state.replace(grid=state.grid.at[
                (state.red_pos[0], state.red_pos[1])
            ].set(jnp.int8(Items.empty)))

            state = state.replace(grid=state.grid.at[
                (state.blue_pos[0], state.blue_pos[1])
            ].set(jnp.int8(Items.empty)))

            state = state.replace(grid=state.grid.at[(new_red_pos[0], new_red_pos[1])].set(
                jnp.int8(Items.red_agent)
            ))

            state = state.replace(grid=state.grid.at[(new_blue_pos[0], new_blue_pos[1])].set(
                jnp.int8(Items.blue_agent)
            ))

            state = state.replace(red_pos=new_red_pos)
            state = state.replace(blue_pos=new_blue_pos)

            red_reward, blue_reward = 0, 0
            (
                interact,
                red_interact_reward,
                blue_interact_reward,
                state,
            ) = _interact(state, (action_0, action_1))
            red_reward += red_interact_reward
            blue_reward += blue_interact_reward

            # if we interacted, then we set freeze
            state = state.replace(freeze=jnp.where(
                interact, freeze_penalty, state.freeze
            ))

            # if we didn't interact, then we decrement freeze
            state = state.replace(freeze=jnp.where(
                state.freeze > 0, state.freeze - 1, state.freeze
            ))
            state_sft_re = _soft_reset_state(key, state)
            state = jax.tree.map(
                lambda x, y: jnp.where(state.freeze == 0, x, y),
                state_sft_re,
                state,
            )
            state_nxt = State(
                red_pos=state.red_pos,
                blue_pos=state.blue_pos,
                inner_t=state.inner_t + 1,
                outer_t=state.outer_t,
                grid=state.grid,
                red_inventory=state.red_inventory,
                blue_inventory=state.blue_inventory,
                red_coins=state.red_coins,
                blue_coins=state.blue_coins,
                freeze=jnp.where(
                    interact, freeze_penalty, state.freeze
                ),
            )

            # now calculate if done for inner or outer episode
            inner_t = state_nxt.inner_t
            outer_t = state_nxt.outer_t
            reset_inner = inner_t == num_inner_steps

            # if inner episode is done, return start state for next game
            state_re = _reset_state(key)
            state_re = state_re.replace(outer_t=outer_t + 1)
            state = jax.tree.map(
                lambda x, y: jax.lax.select(reset_inner, x, y),
                state_re,
                state_nxt,
            )
            outer_t = state.outer_t
            reset_outer = outer_t == num_outer_steps
            done = {}
            done["__all__"] = reset_outer

            obs = _get_obs(state)
            blue_reward = jnp.where(reset_inner, 0, blue_reward)
            red_reward = jnp.where(reset_inner, 0, red_reward)
            return (
                obs,
                state,
                (red_reward, blue_reward),
                done,
                {"discount": jnp.zeros((), dtype=jnp.int8)},
            )

        def _soft_reset_state(key: jnp.ndarray, state: State) -> State:
            """Reset the grid to original state and"""
            # Find the free spaces in the grid
            grid = jnp.zeros((GRID_SIZE, GRID_SIZE), jnp.int8)

            # if coin location can change, then we need to reset the coins
            for i in range(NUM_COINS):
                grid = grid.at[
                    state.red_coins[i, 0], state.red_coins[i, 1]
                ].set(jnp.int8(Items.red_coin))

            for i in range(NUM_COINS):
                grid = grid.at[
                    state.blue_coins[i, 0], state.blue_coins[i, 1]
                ].set(jnp.int8(Items.blue_coin))

            agent_pos = jax.random.choice(
                key, AGENT_SPAWNS, shape=(), replace=False
            )

            player_dir = jax.random.randint(
                key, shape=(2,), minval=0, maxval=3, dtype=jnp.int8
            )
            player_pos = jnp.array(
                [agent_pos[:2, 0], agent_pos[:2, 1], player_dir]
            ).T

            red_pos = player_pos[0, :]
            blue_pos = player_pos[1, :]

            grid = grid.at[red_pos[0], red_pos[1]].set(
                jnp.int8(Items.red_agent)
            )
            grid = grid.at[blue_pos[0], blue_pos[1]].set(
                jnp.int8(Items.blue_agent)
            )

            return State(
                red_pos=red_pos,
                blue_pos=blue_pos,
                inner_t=state.inner_t,
                outer_t=state.outer_t,
                grid=grid,
                red_inventory=jnp.zeros(2),
                blue_inventory=jnp.zeros(2),
                red_coins=state.red_coins,
                blue_coins=state.blue_coins,
                freeze=jnp.int16(-1),
            )

        def _reset_state(
            key: jnp.ndarray
        ) -> Tuple[jnp.ndarray, State]:
            key, subkey = jax.random.split(key)

            # coin_pos = jax.random.choice(
            #     subkey, COIN_SPAWNS, shape=(NUM_COIN_TYPES*NUM_COINS,), replace=False
            # )

            agent_pos = jax.random.choice(
                subkey, AGENT_SPAWNS, shape=(), replace=False
            )
            player_dir = jax.random.randint(
                subkey, shape=(2,), minval=0, maxval=3, dtype=jnp.int8
            )
            player_pos = jnp.array(
                [agent_pos[:2, 0], agent_pos[:2, 1], player_dir]
            ).T
            grid = jnp.zeros((GRID_SIZE, GRID_SIZE), jnp.int8)
            grid = grid.at[player_pos[0, 0], player_pos[0, 1]].set(
                jnp.int8(Items.red_agent)
            )
            grid = grid.at[player_pos[1, 0], player_pos[1, 1]].set(
                jnp.int8(Items.blue_agent)
            )
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
                    jnp.int8(Items.red_coin)
                )

            for i in range(NUM_COINS):
                grid = grid.at[blue_coins[i, 0], blue_coins[i, 1]].set(
                    jnp.int8(Items.blue_coin)
                )

            return State(
                red_pos=player_pos[0, :],
                blue_pos=player_pos[1, :],
                inner_t=0,
                outer_t=0,
                grid=grid,
                red_inventory=jnp.zeros(2),
                blue_inventory=jnp.zeros(2),
                red_coins=red_coins,
                blue_coins=blue_coins,
                freeze=jnp.int16(-1),
            )

        def reset(
            key: jnp.ndarray
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

    @property
    def name(self) -> str:
        """Environment name."""
        return "2PMGinTheGrid"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return len(Actions)

    def action_space(
        self, agent_id: Union[int, None] = None
    ) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(len(Actions))

    def observation_space(self) -> spaces.Dict:
        """Observation space of the environment."""
        _shape = (
            (OBS_SIZE, OBS_SIZE, len(Items) - 1 + 4)
            if self.cnn
            else (OBS_SIZE**2 * (len(Items) - 1 + 4),)
        )

        return {
            "observation": spaces.Box(
                low=0, high=1, shape=_shape, dtype=jnp.uint8
            ),
            "inventory": spaces.Box(
                low=0,
                high=NUM_COINS,
                shape=NUM_COIN_TYPES + 4,
                dtype=jnp.uint8,
            ),
        }

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

        if obj == Items.red_agent:
            # Draw the agent 1
            agent_color = PLAYER1_COLOUR

        elif obj == Items.blue_agent:
            # Draw agent 2
            agent_color = PLAYER2_COLOUR
        elif obj == Items.red_coin:
            # Draw the red coin as GREEN COOPERATE
            fill_coords(
                img, point_in_circle(0.5, 0.5, 0.31), (44.0, 160.0, 44.0)
            )
        elif obj == Items.blue_coin:
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

    def render_agent_view(
        self, state: State, agent: int
    ) -> Tuple[onp.ndarray]:
        """
        Render the observation for each agent"""

        tile_size = 32
        obs = self.get_obs(state)

        grid = onp.array(obs[agent]["observation"][:, :, :-4])
        empty_space_channel = onp.zeros((OBS_SIZE, OBS_SIZE, 1))
        grid = onp.concatenate((empty_space_channel, grid), axis=-1)
        grid = onp.argmax(grid.reshape(-1, grid.shape[-1]), axis=1)
        grid = grid.reshape(OBS_SIZE, OBS_SIZE)

        angles = onp.array(obs[agent]["observation"][:, :, -4:])
        angles = onp.argmax(angles.reshape(-1, angles.shape[-1]), axis=1)
        angles = angles.reshape(OBS_SIZE, OBS_SIZE)

        # Compute the total grid size
        width_px = grid.shape[0] * tile_size
        height_px = grid.shape[0] * tile_size

        img = onp.zeros(shape=(height_px, width_px, 3), dtype=onp.uint8)

        # agent direction
        pricipal_dir = 0
        red_hat = bool(state.red_inventory.sum() > INTERACT_THRESHOLD)
        blue_hat = bool(state.blue_inventory.sum() > INTERACT_THRESHOLD)
        if agent == 0:
            principal_hat = red_hat
            other_hat = blue_hat
        else:
            principal_hat = blue_hat
            other_hat = red_hat

        # Render the grid
        for j in range(0, grid.shape[1]):
            for i in range(0, grid.shape[0]):
                cell = grid[i, j]
                if cell == 0:
                    cell = None

                principal_agent_here = cell == 1
                other_agent_here = cell == 2

                if principal_agent_here:
                    agent_dir = pricipal_dir
                    agent_hat = principal_hat

                elif other_agent_here:
                    agent_dir = angles[i, j]
                    agent_hat = other_hat

                else:
                    agent_dir = None
                    agent_hat = None

                tile_img = InTheGrid_2p.render_tile(
                    cell,
                    agent_dir=agent_dir,
                    agent_hat=agent_hat,
                    highlight=None,
                    tile_size=tile_size,
                )

                ymin = j * tile_size
                ymax = (j + 1) * tile_size
                xmin = i * tile_size
                xmax = (i + 1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img
        img = onp.rot90(img, 2, axes=(0, 1))
        inv = self.render_inventory(
            state.red_inventory if agent == 0 else state.blue_inventory,
            width_px,
        )
        return onp.concatenate((img, inv), axis=0)

    def render(
        self,
        state: State
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
            grid, ((PADDING, PADDING), (PADDING, PADDING)), constant_values=5
        )

        startx, starty = self.get_obs_point(
            state.red_pos[0], state.red_pos[1], state.red_pos[2]
        )
        highlight_mask[
            startx : startx + OBS_SIZE, starty : starty + OBS_SIZE
        ] = True

        startx, starty = self.get_obs_point(
            state.blue_pos[0], state.blue_pos[1], state.blue_pos[2]
        )
        highlight_mask[
            startx : startx + OBS_SIZE, starty : starty + OBS_SIZE
        ] = True
        if state.freeze > 0:
            # check which agent won
            r1, r2 = self.get_reward(state)

            if r1 == -r2:
                # zero sum game
                if r1 > r2:
                    # red won
                    img = onp.tile(
                        PLAYER1_COLOUR, (img.shape[0], img.shape[1], 1)
                    )
                elif r2 > r1:
                    # blue won
                    img = onp.tile(
                        PLAYER2_COLOUR, (img.shape[0], img.shape[1], 1)
                    )
                elif r1 == r2:
                    img[:, : width_px // 2, :] = onp.tile(
                        PLAYER2_COLOUR, (img.shape[0], img.shape[1] // 2, 1)
                    )
                    img[:, width_px // 2 :, :] = onp.tile(
                        PLAYER1_COLOUR, (img.shape[0], img.shape[1] // 2, 1)
                    )
            else:
                # otherwise we got some cool general sum game
                welfare = r1 + r2
                if welfare > 5:
                    # cooperate
                    img[:, width_px // 2 :, :] = onp.tile(
                        GREEN_COLOUR, (img.shape[0], img.shape[1] // 2, 1)
                    )
                else:
                    img[:, width_px // 2 :, :] = onp.tile(
                        RED_COLOUR, (img.shape[0], img.shape[1] // 2, 1)
                    )
                if r1 > r2:
                    # red won
                    img[:, : width_px // 2, :] = onp.tile(
                        PLAYER1_COLOUR, (img.shape[0], img.shape[1] // 2, 1)
                    )
                elif r1 < r2:
                    # blue won
                    img[:, : width_px // 2, :] = onp.tile(
                        PLAYER2_COLOUR, (img.shape[0], img.shape[1] // 2, 1)
                    )
                elif r1 == r2:
                    img[height_px // 2 :, : width_px // 2, :] = onp.tile(
                        PLAYER1_COLOUR,
                        (img.shape[0] // 2, img.shape[1] // 2, 1),
                    )
                    img[: height_px // 2, : width_px // 2, :] = onp.tile(
                        PLAYER2_COLOUR,
                        (img.shape[0] // 2, img.shape[1] // 2, 1),
                    )

            img = img.astype(onp.uint8)
        else:
            # Render the grid
            for j in range(0, grid.shape[1]):
                for i in range(0, grid.shape[0]):
                    cell = grid[i, j]
                    if cell == 0:
                        cell = None
                    red_agent_here = cell == 1
                    blue_agent_here = cell == 2

                    agent_dir = None
                    agent_dir = (
                        state.red_pos[2].item()
                        if red_agent_here
                        else agent_dir
                    )
                    agent_dir = (
                        state.blue_pos[2].item()
                        if blue_agent_here
                        else agent_dir
                    )
                    agent_hat = False
                    agent_hat = (
                        bool(state.red_inventory.sum() > INTERACT_THRESHOLD)
                        if red_agent_here
                        else agent_hat
                    )
                    agent_hat = (
                        bool(state.blue_inventory.sum() > INTERACT_THRESHOLD)
                        if blue_agent_here
                        else agent_hat
                    )

                    tile_img = InTheGrid_2p.render_tile(
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
        red_inv = self.render_inventory(state.red_inventory, img.shape[1])
        blue_inv = self.render_inventory(state.blue_inventory, img.shape[1])

        time = self.render_time(state, img.shape[1])
        img = onp.concatenate((img, red_inv, blue_inv, time), axis=0)
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
                tile_img = InTheGrid_2p.render_tile(cell, tile_size=tile_height)
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