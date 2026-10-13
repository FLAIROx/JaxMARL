import jax
import jax.numpy as jnp

from jaxmarl.environments.overcooked_v2.common import Direction, Position
from jaxmarl.environments.overcooked_v2.layouts import Layout
from jaxmarl.environments.overcooked_v2.overcooked import OvercookedV2


def _walkable_border_layout():
    return Layout.from_string(
        """
WWWWW
A   0
WWWWW
""",
        possible_recipes=[[0, 0, 0]],
    )


def test_interact_from_walkable_left_edge_is_a_no_op():
    env = OvercookedV2(layout=_walkable_border_layout())
    _, state = env.reset(jax.random.PRNGKey(0))

    # Agent starts at the left edge (x=0); facing left, "in front" is off the grid.
    state = state.replace(agents=state.agents.replace(dir=jnp.array([Direction.LEFT])))

    _, next_state, reward, _, _ = env.step_env(
        jax.random.PRNGKey(1), state, {"agent_0": jnp.array(5)}
    )

    assert jnp.all(next_state.agents.inventory == 0)
    assert jnp.all(next_state.grid == state.grid)
    assert reward["agent_0"] == 0


def test_interact_from_walkable_right_edge_is_a_no_op():
    env = OvercookedV2(layout=_walkable_border_layout())
    _, state = env.reset(jax.random.PRNGKey(0))

    # Place the agent at the right edge (x=4), facing right and off the grid.
    state = state.replace(
        agents=state.agents.replace(
            pos=Position(x=jnp.array([4]), y=jnp.array([1])),
            dir=jnp.array([Direction.RIGHT]),
        )
    )

    _, next_state, reward, _, _ = env.step_env(
        jax.random.PRNGKey(1), state, {"agent_0": jnp.array(5)}
    )

    assert jnp.all(next_state.agents.inventory == 0)
    assert jnp.all(next_state.grid == state.grid)
    assert reward["agent_0"] == 0


def test_interact_still_works_next_to_an_in_bounds_pile():
    env = OvercookedV2(layout=_walkable_border_layout())
    _, state = env.reset(jax.random.PRNGKey(0))

    # Stand right next to the ingredient pile (at x=4) and face it.
    state = state.replace(
        agents=state.agents.replace(
            pos=Position(x=jnp.array([3]), y=jnp.array([1])),
            dir=jnp.array([Direction.RIGHT]),
        )
    )

    _, next_state, *_ = env.step_env(
        jax.random.PRNGKey(1), state, {"agent_0": jnp.array(5)}
    )

    assert next_state.agents.inventory[0] != 0
