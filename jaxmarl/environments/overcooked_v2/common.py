import numpy as np
import jax.numpy as jnp
import chex
import jax
from enum import IntEnum
from jax.typing import ArrayLike

MAX_INGREDIENTS = 3


class StaticObject(IntEnum):
    EMPTY = 0
    WALL = 1

    # Agents are only included in the observation grid
    AGENT = 2
    SELF_AGENT = 3

    GOAL = 4
    POT = 5
    RECIPE_INDICATOR = 6
    BUTTON_RECIPE_INDICATOR = 7

    PLATE_PILE = 9
    INGREDIENT_PILE_BASE = 10

    @staticmethod
    def is_ingredient_pile(obj):
        return obj >= StaticObject.INGREDIENT_PILE_BASE

    @staticmethod
    def get_ingredient(obj):
        idx = obj - StaticObject.INGREDIENT_PILE_BASE
        return DynamicObject.ingredient(idx)

    @staticmethod
    def ingredient_pile(idx):
        return StaticObject.INGREDIENT_PILE_BASE + idx


class DynamicObject(IntEnum):
    EMPTY = 0
    PLATE = 1 << 0
    COOKED = 1 << 1

    # every ingredient has two unique bit
    BASE_INGREDIENT = 1 << 2

    @staticmethod
    def ingredient(idx):
        return DynamicObject.BASE_INGREDIENT << 2 * idx

    @staticmethod
    def is_ingredient(obj):
        return ((obj >> 2) != 0) & ((obj & DynamicObject.PLATE) == 0)

    @staticmethod
    def ingredient_count(obj):
        initial_val = (obj >> 2, jnp.array(0))

        def _count_ingredients(x):
            obj, count = x
            return (obj >> 2, count + (obj & 0x3))

        _, count = jax.lax.while_loop(
            lambda x: x[0] > 0, _count_ingredients, initial_val
        )
        return count

    @staticmethod
    def get_ingredient_idx_list(obj):
        res = []
        obj >>= 2
        idx = 0
        while obj > 0:
            res += [idx] * (obj & 0x3).item()
            obj >>= 2
            idx += 1
        return res

    @staticmethod
    def get_ingredient_idx_list_jit(obj):

        def _loop_body(carry):
            obj, pos, idx, res = carry
            count = obj & 0x3

            cond = jnp.arange(MAX_INGREDIENTS)
            cond = (cond >= pos) & (res == -1) & (cond < pos + count)

            res = jnp.where(
                cond,
                idx,
                res,
            )

            return (obj >> 2, pos + count, idx + 1, res)

        def _loop_cond(carry):
            obj, pos, _, _ = carry
            return (obj > 0) & (pos < MAX_INGREDIENTS)

        initial_res = jnp.full((MAX_INGREDIENTS,), -1, dtype=jnp.int32)
        carry = (obj >> 2, 0, 0, initial_res)

        val = jax.lax.while_loop(_loop_cond, _loop_body, carry)
        return val[-1]

    @staticmethod
    def get_ingredient_idx(obj):

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
        ingredients = jax.vmap(DynamicObject.ingredient)(recipe)
        return jnp.sum(ingredients)


class Direction(IntEnum):
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
        (0, -1),
        (0, 1),
        (1, 0),
        (-1, 0),
    ],
    dtype=jnp.int8,
)


@chex.dataclass
class Position:
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
    pos: Position
    dir: jnp.ndarray
    inventory: jnp.ndarray

    def get_fwd_pos(self):
        return self.pos.move(self.dir)

    @staticmethod
    def from_position(pos):
        return Agent(pos, jnp.array([Direction.UP]), jnp.zeros((1,)))


class Actions(IntEnum):
    # Turn left, turn right, move forward
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
