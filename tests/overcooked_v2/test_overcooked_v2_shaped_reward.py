"""Plate pickup shaping in OvercookedV2 when several agents interact in the same step.

Agents interact in index order, each on the grid the previous ones left behind. The plate pickup reward
("fewer plates in hands than non-empty pots") must then count the plates the earlier agents picked up in
the same step too. The same fix was made for Overcooked (v1) in #195.
"""

import jax
import jax.numpy as jnp
import pytest

from jaxmarl.environments.overcooked_v2.common import Direction, DynamicObject
from jaxmarl.environments.overcooked_v2.layouts import Layout
from jaxmarl.environments.overcooked_v2.overcooked import OvercookedV2
from jaxmarl.environments.overcooked_v2.settings import SHAPED_REWARDS

INTERACT = 5
PLATE_PICKUP = SHAPED_REWARDS["PLATE_PICKUP"]
DISH_PICKUP = SHAPED_REWARDS["DISH_PICKUP"]
ONION = DynamicObject.ingredient(0)


def _env_and_state(layout, inventories):
    """All agents face up, holding `inventories`."""
    env = OvercookedV2(layout=Layout.from_string(layout, possible_recipes=[[0, 0, 0]]))
    _, state = env.reset(jax.random.PRNGKey(0))
    agents = state.agents.replace(
        dir=jnp.full_like(state.agents.dir, Direction.UP),
        inventory=jnp.array(inventories, dtype=state.agents.inventory.dtype),
    )
    return env, state.replace(agents=agents)


def _interact_all(env, state):
    actions = {agent: jnp.array(INTERACT) for agent in env.agents}
    _, state, _, _, info = env.step_env(jax.random.PRNGKey(1), state, actions)
    return state, [float(info["shaped_reward"][agent]) for agent in env.agents]


def test_one_pot_rewards_one_plate():
    # Both agents grab a plate from the piles above them; only one pot (bottom middle) needs a plate.
    env, state = _env_and_state("\nWBWBW\nWA AW\nW0PWW\n", [0, 0])
    state = state.replace(grid=state.grid.at[2, 2, 1].set(ONION))

    _, shaped = _interact_all(env, state)

    assert shaped == [PLATE_PICKUP, 0.0]


def test_two_pots_reward_two_of_three_plates():
    env, state = _env_and_state("\nWBWBWBW\nWA A AW\nW0PWPWW\n", [0, 0, 0])
    state = state.replace(grid=state.grid.at[2, 2, 1].set(ONION).at[2, 4, 1].set(ONION))

    _, shaped = _interact_all(env, state)

    assert shaped == [PLATE_PICKUP, PLATE_PICKUP, 0.0]


@pytest.mark.parametrize("dish_agent", [0, 1])
def test_plate_pickup_does_not_depend_on_agent_order(dish_agent):
    # One agent turns its plate into a dish from a cooked pot; the other grabs a bare plate for the
    # partly filled pot. After the step one pot needs a plate and nobody holds one, whichever agent
    # took the dish.
    layout = "\nWPWBW\nWA AW\nW0PWW\n" if dish_agent == 0 else "\nWBWPW\nWA AW\nW0PWW\n"
    inventories = [0, 0]
    inventories[dish_agent] = int(DynamicObject.PLATE)
    env, state = _env_and_state(layout, inventories)
    cooked_x = 1 if dish_agent == 0 else 3
    grid = state.grid.at[0, cooked_x, 1].set(state.recipe | DynamicObject.COOKED)
    state = state.replace(grid=grid.at[2, 2, 1].set(ONION))

    _, shaped = _interact_all(env, state)

    assert shaped[dish_agent] == DISH_PICKUP
    assert shaped[1 - dish_agent] == PLATE_PICKUP
