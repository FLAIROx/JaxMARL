import jax
import jax.numpy as jnp

from jaxmarl.environments.overcooked_v2.common import Direction
from jaxmarl.environments.overcooked_v2.layouts import Layout
from jaxmarl.environments.overcooked_v2.overcooked import OvercookedV2


def _step_facing(layout_string, direction):
    layout = Layout.from_string(layout_string, possible_recipes=[[0, 0, 0]])
    env = OvercookedV2(layout=layout)
    _, state = env.reset(jax.random.PRNGKey(0))
    state = state.replace(agents=state.agents.replace(dir=jnp.array([direction])))
    _, state, *_ = env.step_env(jax.random.PRNGKey(1), state, {"agent_0": jnp.array(5)})
    return state


def test_interact_facing_out_of_walkable_left_edge_is_noop():
    # Agent at the left edge (x=0), onion pile at the far right (x=4).
    # Facing LEFT used to wrap the index to x=-1 -> x=4 and pick up an onion.
    state = _step_facing("\nWWWWW\nA   0\nWWWWW\n", Direction.LEFT)
    assert state.agents.inventory[0] == 0


def test_interact_facing_out_of_walkable_right_edge_is_noop():
    # Agent at the right edge (x=4), onion pile at the far left (x=0).
    # Facing RIGHT pushes the index to x=5, which JAX clips in-bounds instead
    # of wrapping, but must still be treated as a no-op interaction.
    state = _step_facing("\nWWWWW\n0   A\nWWWWW\n", Direction.RIGHT)
    assert state.agents.inventory[0] == 0
