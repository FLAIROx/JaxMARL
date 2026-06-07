"""Common data structures and enums for Overcooked V3."""

import jax
import jax.numpy as jnp
import chex
from enum import IntEnum
from jax.typing import ArrayLike

MAX_INGREDIENTS = 3


class StaticObject(IntEnum):
    """Static objects on the grid (channel 0)."""
    EMPTY = 0
    WALL = 1

    # Agents are only included in the observation grid
    AGENT = 2
    SELF_AGENT = 3

    GOAL = 4
    POT = 5
    RECIPE_INDICATOR = 6

    PLATE_PILE = 9
    INGREDIENT_PILE_BASE = 10

    # Conveyor belts - directions encoded in extra channel
    ITEM_CONVEYOR = 20
    PLAYER_CONVEYOR = 21

    # Moving walls and buttons
    MOVING_WALL = 22
    BUTTON = 23
    BARRIER = 24

    @staticmethod
    def is_ingredient_pile(obj):
        return (obj >= StaticObject.INGREDIENT_PILE_BASE) & (obj < StaticObject.ITEM_CONVEYOR)

    @staticmethod
    def get_ingredient(obj):
        idx = obj - StaticObject.INGREDIENT_PILE_BASE
        return DynamicObject.ingredient(idx)

    @staticmethod
    def ingredient_pile(idx):
        return StaticObject.INGREDIENT_PILE_BASE + idx


class ButtonAction(IntEnum):
    """Actions a button can trigger on its linked moving wall or barrier."""
    TOGGLE_PAUSE = 0       # Pause/unpause the wall's movement
    TOGGLE_DIRECTION = 1   # Reverse the wall's direction
    TOGGLE_BOUNCE = 2      # Toggle bounce mode on/off
    TRIGGER_MOVE = 3       # Move the wall one step (wall is paused by default)
    
    # Barrier actions (button target indexes refer to barriers instead of moving walls)
    TOGGLE_BARRIER = 4     # Toggle barrier active state
    TIMED_BARRIER = 5      # Deactivate barrier temporarily (auto-reactivates)


class DynamicObject(IntEnum):
    """Dynamic objects (channel 1) - bitwise encoding."""
    EMPTY = 0
    PLATE = 1 << 0        # bit 0: plate
    COOKED = 1 << 1       # bit 1: cooked flag

    # Every ingredient has two bits (count 0-3)
    BASE_INGREDIENT = 1 << 2

    # Burning flags (bits 6-7)
    BURNING = 1 << 6      # Pot is in burning window
    BURNED = 1 << 7       # Pot has burned (contents destroyed)

    @staticmethod
    def ingredient(idx):
        """Get the bit pattern for a single ingredient of given type."""
        return DynamicObject.BASE_INGREDIENT << (2 * idx)

    @staticmethod
    def is_ingredient(obj):
        """Check if object contains only ingredients (no plate)."""
        return ((obj >> 2) != 0) & ((obj & DynamicObject.PLATE) == 0)

    @staticmethod
    def ingredient_count(obj):
        """Count total number of ingredients in the object."""
        initial_val = (obj >> 2, jnp.array(0))

        def _count_ingredients(x):
            obj, count = x
            return (obj >> 2, count + (obj & 0x3))

        _, count = jax.lax.while_loop(
            lambda x: x[0] > 0, _count_ingredients, initial_val
        )
        return count

    @staticmethod
    def get_ingredient_type(obj):
        """Get the type index of ingredients in the object (assumes single type)."""
        def _body_fun(val):
            obj, idx, res = val
            new_res = jax.lax.select(obj & 0x3 != 0, idx, res)
            return (obj >> 2, idx + 1, new_res)

        def _cond_fun(val):
            obj, _, res = val
            return (obj > 0) & (res == -1)

        initial_val = (obj >> 2, 0, -1)
        val = jax.lax.while_loop(_cond_fun, _body_fun, initial_val)
        return val[-1]

    @staticmethod
    def get_recipe_encoding(recipe: ArrayLike):
        """Encode a recipe (list of ingredient indices) into a single int."""
        ingredients = jax.vmap(DynamicObject.ingredient)(recipe)
        return jnp.sum(ingredients)

    @staticmethod
    def is_soup(obj):
        """Check if object is a cooked soup on a plate."""
        return ((obj & DynamicObject.COOKED) != 0) & ((obj & DynamicObject.PLATE) != 0)


class Direction(IntEnum):
    """Cardinal directions."""
    UP = 0
    DOWN = 1
    RIGHT = 2
    LEFT = 3

    @staticmethod
    def opposite(dir):
        opposite_map = jnp.array(
            [Direction.DOWN, Direction.UP, Direction.LEFT, Direction.RIGHT]
        )
        return opposite_map[dir]


ALL_DIRECTIONS = jnp.array(
    [Direction.UP, Direction.DOWN, Direction.RIGHT, Direction.LEFT]
)

DIR_TO_VEC = jnp.array(
    [
        (0, -1),   # UP
        (0, 1),    # DOWN
        (1, 0),    # RIGHT
        (-1, 0),   # LEFT
    ],
    dtype=jnp.int8,
)


@chex.dataclass
class Position:
    """2D position on the grid."""
    x: jnp.ndarray
    y: jnp.ndarray

    @staticmethod
    def from_tuple(t):
        x, y = t
        return Position(jnp.array([x]), jnp.array([y]))

    def move(self, direction):
        vec = DIR_TO_VEC[direction]
        return Position(x=self.x + vec[0], y=self.y + vec[1])

    def move_in_bounds(self, direction, width, height):
        new_pos = self.move(direction)
        clipped_x = jnp.clip(new_pos.x, 0, width - 1)
        clipped_y = jnp.clip(new_pos.y, 0, height - 1)
        return Position(x=clipped_x, y=clipped_y)

    def checked_move(self, direction, width, height):
        new_pos = self.move(direction)
        clipped_x = jnp.clip(new_pos.x, 0, width - 1)
        clipped_y = jnp.clip(new_pos.y, 0, height - 1)
        return Position(x=clipped_x, y=clipped_y), (clipped_x == new_pos.x) & (
            clipped_y == new_pos.y
        )

    def to_array(self):
        return jnp.stack([self.x, self.y], axis=-1)

    def delta(self, other):
        return jnp.array([self.x - other.x, self.y - other.y])


@chex.dataclass
class Agent:
    """Agent state."""
    pos: Position
    dir: jnp.ndarray
    inventory: jnp.ndarray

    def get_fwd_pos(self):
        return self.pos.move(self.dir)

    @staticmethod
    def from_position(pos):
        return Agent(pos, jnp.array([Direction.UP]), jnp.zeros((1,)))


class Actions(IntEnum):
    """Available agent actions."""
    right = 0
    down = 1
    left = 2
    up = 3
    stay = 4
    interact = 5


ACTION_TO_DIRECTION = (
    jnp.full((len(Actions),), -1)
    .at[Actions.right]
    .set(Direction.RIGHT)
    .at[Actions.down]
    .set(Direction.DOWN)
    .at[Actions.left]
    .set(Direction.LEFT)
    .at[Actions.up]
    .set(Direction.UP)
)


# Soup types for order queue
class SoupType(IntEnum):
    """Types of soup that can be ordered."""
    NONE = 0
    ONION_SOUP = 1
    TOMATO_SOUP = 2
