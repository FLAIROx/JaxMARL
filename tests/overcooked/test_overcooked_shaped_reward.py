"""Shaped reward tests for Overcooked (v1).

Regression coverage for https://github.com/FLAIROx/JaxMARL/issues/167, where the
plate pickup shaping never fired because walls were miscounted as plates, and
plates were only counted in the acting agent's own inventory.
"""

import jax
import jax.numpy as jnp
import pytest

from jaxmarl.environments.overcooked.common import OBJECT_TO_INDEX
from jaxmarl.environments.overcooked.layouts import overcooked_layouts
from jaxmarl.environments.overcooked.overcooked import (
    BASE_REW_SHAPING_PARAMS,
    Overcooked,
)

PLATE_PICKUP_REWARD = BASE_REW_SHAPING_PARAMS["PLATE_PICKUP_REWARD"]


@pytest.fixture
def env():
    return Overcooked(layout=overcooked_layouts["cramped_room"])


@pytest.fixture
def state(env):
    _, state = env.reset(jax.random.PRNGKey(0))
    return state


def _padding(env, state):
    return (state.maze_map.shape[0] - env.obs_shape[1]) // 2


def _find(env, state, obj):
    """(y, x) of the first cell holding `obj`, in unpadded coordinates."""
    pad = _padding(env, state)
    return jnp.argwhere(
        state.maze_map[pad:-pad, pad:-pad, 0] == OBJECT_TO_INDEX[obj], size=1
    )[0]


def _pickup_plate(env, state, maze_map, inventory_all):
    """Agent 0 interacts with the plate pile, agent 1 stands elsewhere."""
    plate_pile = _find(env, state, "plate_pile")
    fwd_pos_all = jnp.array([[plate_pile[1], plate_pile[0]], [0, 0]], dtype=jnp.uint32)
    return env.process_interact(maze_map, state.wall_map, fwd_pos_all, inventory_all, 0)


def _fill_pot(env, state, status):
    pad = _padding(env, state)
    pot_pos = state.pot_pos[0]
    return state.maze_map.at[pad + pot_pos[1], pad + pot_pos[0], 2].set(status)


def test_plate_pickup_is_rewarded(env, state):
    """A useful plate pickup earns the shaping reward.

    Regression for the wall/plate index collision: `plate` and `grey` are both
    index 5, so comparing the whole 3D maze_map made every wall look like a
    plate on a counter, permanently disabling this reward.
    """
    maze_map = _fill_pot(env, state, 19)  # pot is cooking, so a plate is useful
    empty = jnp.full((2,), OBJECT_TO_INDEX["empty"], dtype=jnp.uint8)

    _, inventory, _, shaped_reward = _pickup_plate(env, state, maze_map, empty)

    assert inventory == OBJECT_TO_INDEX["plate"]
    assert shaped_reward == PLATE_PICKUP_REWARD


def test_plate_pickup_not_rewarded_when_no_pot_is_in_use(env, state):
    """With all pots empty there is nothing to plate, so no reward."""
    empty = jnp.full((2,), OBJECT_TO_INDEX["empty"], dtype=jnp.uint8)

    _, inventory, _, shaped_reward = _pickup_plate(env, state, state.maze_map, empty)

    assert inventory == OBJECT_TO_INDEX["plate"]
    assert shaped_reward == 0.0


def test_plate_pickup_not_rewarded_when_other_agent_holds_a_plate(env, state):
    """Plates are counted across all agents, not just the acting one.

    One cooking pot only needs one plate; if the other agent already carries
    one, a second pickup is not useful.
    """
    maze_map = _fill_pot(env, state, 19)
    other_holds_plate = jnp.array(
        [OBJECT_TO_INDEX["empty"], OBJECT_TO_INDEX["plate"]], dtype=jnp.uint8
    )

    _, _, _, shaped_reward = _pickup_plate(env, state, maze_map, other_holds_plate)

    assert shaped_reward == 0.0


def test_plate_pickup_not_rewarded_when_a_plate_sits_on_a_counter(env, state):
    """Discourages picking a plate up and dropping it repeatedly."""
    pad = _padding(env, state)
    maze_map = _fill_pot(env, state, 19)

    # Drop a plate onto an empty counter (a wall cell that is not the plate pile)
    counter = jnp.argwhere(
        maze_map[pad:-pad, pad:-pad, 0] == OBJECT_TO_INDEX["wall"], size=1
    )[0]
    maze_map = maze_map.at[pad + counter[0], pad + counter[1], 0].set(
        OBJECT_TO_INDEX["plate"]
    )
    empty = jnp.full((2,), OBJECT_TO_INDEX["empty"], dtype=jnp.uint8)

    _, _, _, shaped_reward = _pickup_plate(env, state, maze_map, empty)

    assert shaped_reward == 0.0
