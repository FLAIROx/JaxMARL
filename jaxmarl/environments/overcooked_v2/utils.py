from functools import partial
import jax
import jax.numpy as jnp
from typing import List, Tuple
import itertools
from .common import ALL_DIRECTIONS, Position, Direction


def tree_select(predicate, a, b):
    return jax.tree_util.tree_map(lambda x, y: jax.lax.select(predicate, x, y), a, b)


def compute_view_box(x, y, agent_view_size, height, width):
    """Compute the view box for an agent centered at (x, y)"""
    x_low = x - agent_view_size
    x_high = x + agent_view_size + 1
    y_low = y - agent_view_size
    y_high = y + agent_view_size + 1

    x_low = jax.lax.clamp(0, x_low, width)
    x_high = jax.lax.clamp(0, x_high, width)
    y_low = jax.lax.clamp(0, y_low, height)
    y_high = jax.lax.clamp(0, y_high, height)

    return x_low, x_high, y_low, y_high


def compute_enclosed_spaces(empty_mask: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the enclosed spaces in the environment.
    Each enclosed space is assigned a unique id.
    """
    height, width = empty_mask.shape
    id_grid = jnp.arange(empty_mask.size, dtype=jnp.int32).reshape(empty_mask.shape)
    id_grid = jnp.where(empty_mask, id_grid, -1)

    def _body_fun(val):
        _, curr = val

        def _next_val(pos):
            neighbors = jax.vmap(pos.move_in_bounds, in_axes=(0, None, None))(
                jnp.array(list(Direction)), width, height
            )
            neighbour_values = curr[neighbors.y, neighbors.x]
            self_value = curr[pos.y, pos.x]
            values = jnp.concatenate(
                [neighbour_values, self_value[jnp.newaxis]], axis=0
            )
            new_val = jnp.max(values)
            return jax.lax.select(self_value == -1, self_value, new_val)

        pos_y, pos_x = jnp.meshgrid(
            jnp.arange(height), jnp.arange(width), indexing="ij"
        )

        next_vals = jax.vmap(jax.vmap(_next_val))(Position(x=pos_x, y=pos_y))
        stop = jnp.all(curr == next_vals)
        return stop, next_vals

    def _cond_fun(val):
        return ~val[0]

    initial_val = (False, id_grid)
    _, res = jax.lax.while_loop(_cond_fun, _body_fun, initial_val)
    return res


def mark_adjacent_cells(mask):
    # Shift the mask in four directions: up, down, left, right
    up = jnp.roll(mask, shift=-1, axis=0)
    down = jnp.roll(mask, shift=1, axis=0)
    left = jnp.roll(mask, shift=-1, axis=1)
    right = jnp.roll(mask, shift=1, axis=1)

    # Prevent wrapping by zeroing out the rolled values at the boundaries
    up = up.at[-1, :].set(False)
    down = down.at[0, :].set(False)
    left = left.at[:, -1].set(False)
    right = right.at[:, 0].set(False)

    # Combine the original mask with the shifted versions
    expanded_mask = mask | up | down | left | right

    return expanded_mask


def get_closest_true_pos_no_directions(arr: jnp.ndarray, pos: Position) -> Position:
    height, width = arr.shape

    y, x = pos.y, pos.x
    yy, xx = jnp.meshgrid(jnp.arange(height), jnp.arange(width), indexing="ij")

    dist = jnp.abs(yy - y) + jnp.abs(xx - x)
    dist = jnp.where(arr, dist, jnp.inf)

    min_idx = jnp.argmin(dist)
    min_y, min_x = jnp.divmod(min_idx, width)

    is_valid = jnp.any(arr)

    return Position(x=min_x, y=min_y), is_valid


class OvercookedPathPlanner:
    def __init__(self, move_area: jnp.ndarray):
        self._precompute(move_area)

    @partial(jax.jit, static_argnums=(0,))
    def get_closest_target_pos(
        self, targets: jnp.ndarray, pos: Position, dir: Direction
    ) -> Tuple[Position, bool]:
        pos_lookup_idx = self.position_to_idx[pos.y, pos.x]

        def _compute_min_moves():
            min_moves = self.precomputed_min_moves[pos_lookup_idx, dir]
            return self._get_pos_from_min_moves_grid(min_moves, targets)

        is_target = targets[pos.y, pos.x]
        is_allowed_pos = pos_lookup_idx != -1
        return jax.lax.cond(
            is_target | ~is_allowed_pos,
            lambda: (pos, is_allowed_pos),
            _compute_min_moves,
        )

    def _precompute(self, move_area: jnp.ndarray):
        pos_idx = jnp.argwhere(move_area)
        num_pos = pos_idx.shape[0]
        positions = Position(y=pos_idx[:, 0], x=pos_idx[:, 1])

        self.precomputed_min_moves = jax.vmap(
            jax.vmap(self._compute_min_moves, in_axes=(None, 0, None)),
            in_axes=(0, None, None),
        )(positions, ALL_DIRECTIONS, move_area)

        self.position_to_idx = (
            jnp.full_like(move_area, -1, dtype=jnp.int32)
            .at[positions.y, positions.x]
            .set(jnp.arange(num_pos))
        )

        # print("Precomputation done")
        # print("Number of positions:", num_pos)
        # print("Shape of precomputed_min_moves:", self.precomputed_min_moves.shape)
        # print("Shape of position_to_idx:", self.position_to_idx.shape)

    @classmethod
    @partial(jax.jit, static_argnums=(0,))
    def get_closest_target_pos_static(
        cls, move_area: jnp.ndarray, targets: jnp.ndarray, pos: Position, dir: Direction
    ) -> Tuple[Position, bool]:

        def _compute_min_moves(pos, dir):
            min_moves = cls._compute_min_moves(pos, dir, move_area)
            return cls._get_pos_from_min_moves_grid(min_moves, targets)

        return jax.lax.cond(
            targets[pos.y, pos.x],
            lambda p, _d: (p, True),
            _compute_min_moves,
            pos,
            dir,
        )

    @staticmethod
    def _get_pos_from_min_moves_grid(
        min_moves: jnp.ndarray, targets: jnp.ndarray
    ) -> Tuple[Position, bool]:
        min_moves_targets = jnp.where(targets, min_moves, jnp.inf)

        min_idx = jnp.argmin(min_moves_targets)
        min_y, min_x = jnp.divmod(min_idx, min_moves.shape[1])

        is_valid = jnp.any(jnp.isfinite(min_moves_targets))

        return Position(x=min_x, y=min_y), is_valid

    @staticmethod
    @jax.jit
    def _compute_min_moves(
        pos: Position, dir: Direction, mask: jnp.ndarray
    ) -> jnp.ndarray:
        assert mask.ndim == 2 and mask.dtype == jnp.bool_
        H, W = mask.shape

        ys, xs = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing="ij")

        def _obstacle_ahead(pos, dir):
            new_pos, is_valid = pos.checked_move(dir, W, H)
            obstacle_ahead = ~mask[new_pos.y, new_pos.x]
            return ~is_valid | obstacle_ahead

        obstacle_ahead = jax.vmap(_obstacle_ahead, in_axes=(None, 0), out_axes=-1)(
            Position(x=xs, y=ys), ALL_DIRECTIONS
        )
        obstacle_ahead &= mask[..., jnp.newaxis]

        def cond_fun(loop_carry):
            _, changed = loop_carry
            return changed

        def body_fun(loop_carry):
            dist, _ = loop_carry
            changed = False

            min_across_last_dim = jnp.min(dist, axis=-1)

            def _move_dir(dir):
                moved_dist = jnp.full_like(min_across_last_dim, jnp.inf)
                if dir == Direction.UP:
                    moved_dist = moved_dist.at[:-1, :].set(min_across_last_dim[1:, :])
                elif dir == Direction.DOWN:
                    moved_dist = moved_dist.at[1:, :].set(min_across_last_dim[:-1, :])
                elif dir == Direction.RIGHT:
                    moved_dist = moved_dist.at[:, 1:].set(min_across_last_dim[:, :-1])
                elif dir == Direction.LEFT:
                    moved_dist = moved_dist.at[:, :-1].set(min_across_last_dim[:, 1:])
                return jnp.where(mask, moved_dist, jnp.inf)

            dist_up = _move_dir(Direction.UP)
            dist_down = _move_dir(Direction.DOWN)
            dist_right = _move_dir(Direction.RIGHT)
            dist_left = _move_dir(Direction.LEFT)

            dist_new = jnp.stack([dist_up, dist_down, dist_right, dist_left], axis=-1)

            blocked_new_dist = jnp.where(
                obstacle_ahead, min_across_last_dim[..., jnp.newaxis], jnp.inf
            )

            dist_new = jnp.minimum(dist_new, blocked_new_dist)
            dist_new += 1
            dist_updated = jnp.minimum(dist, dist_new)

            changed = jnp.any(dist_updated != dist)

            return dist_updated, changed

        initial_dist = jnp.full((H, W, 4), jnp.inf, dtype=jnp.float32)
        initial_dist = initial_dist.at[pos.y, pos.x, dir].set(0)

        dist_final, _ = jax.lax.while_loop(cond_fun, body_fun, (initial_dist, True))

        def _compute_min_cost(pos, dir):
            new_pos, is_valid = pos.checked_move(dir, W, H)
            opposite_dir = Direction.opposite(dir)

            return jnp.where(
                is_valid,
                dist_final[new_pos.y, new_pos.x, opposite_dir],
                jnp.inf,
            )

        min_cost_to_target = jax.vmap(
            _compute_min_cost, in_axes=(None, 0), out_axes=-1
        )(Position(x=xs, y=ys), ALL_DIRECTIONS)
        min_cost_to_target = jnp.min(min_cost_to_target, axis=-1)

        # we only care about target cells
        min_cost_to_target = jnp.where(mask, jnp.inf, min_cost_to_target)

        return min_cost_to_target
