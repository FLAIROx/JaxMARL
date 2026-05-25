"""Visualizer for Overcooked V3 environment."""

import math
from functools import partial
import jax
import jax.numpy as jnp

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

from jaxmarl.viz.window import Window
import jaxmarl.viz.grid_rendering_v2 as rendering
from jaxmarl.environments.overcooked_v3.common import StaticObject, DynamicObject
from jaxmarl.environments.overcooked_v3.settings import POT_COOK_TIME, POT_BURN_TIME

TILE_PIXELS = 32

COLORS = {
    "red": jnp.array([255, 0, 0], dtype=jnp.uint8),
    "green": jnp.array([0, 255, 0], dtype=jnp.uint8),
    "blue": jnp.array([0, 0, 255], dtype=jnp.uint8),
    "purple": jnp.array([160, 32, 240], dtype=jnp.uint8),
    "yellow": jnp.array([255, 255, 0], dtype=jnp.uint8),
    "grey": jnp.array([100, 100, 100], dtype=jnp.uint8),
    "white": jnp.array([255, 255, 255], dtype=jnp.uint8),
    "black": jnp.array([25, 25, 25], dtype=jnp.uint8),
    "orange": jnp.array([230, 180, 0], dtype=jnp.uint8),
    "pink": jnp.array([255, 105, 180], dtype=jnp.uint8),
    "brown": jnp.array([139, 69, 19], dtype=jnp.uint8),
    "cyan": jnp.array([0, 255, 255], dtype=jnp.uint8),
    "light_blue": jnp.array([173, 216, 230], dtype=jnp.uint8),
    "dark_green": jnp.array([0, 150, 0], dtype=jnp.uint8),
    "light_grey": jnp.array([180, 180, 180], dtype=jnp.uint8),
}

INGREDIENT_COLORS = jnp.array(
    [
        COLORS["yellow"],      # Onion
        COLORS["red"],         # Tomato
        COLORS["dark_green"],  # Lettuce
        COLORS["cyan"],
        COLORS["orange"],
        COLORS["purple"],
        COLORS["blue"],
        COLORS["pink"],
        COLORS["brown"],
        COLORS["white"],
    ]
)

AGENT_COLORS = jnp.array(
    [
        COLORS["blue"],
        COLORS["green"],
        COLORS["red"],
        COLORS["purple"],
        COLORS["yellow"],
        COLORS["orange"],
    ]
)


class OvercookedV3Visualizer:
    """Visualizer for Overcooked V3 environment."""

    tile_cache = {}

    def __init__(self, env, tile_size=TILE_PIXELS, subdivs=3):
        self.env = env
        self.window = None
        self.tile_size = tile_size
        self.subdivs = subdivs
        self.pot_cook_time = getattr(env, 'pot_cook_time', POT_COOK_TIME)
        self.pot_burn_time = getattr(env, 'pot_burn_time', POT_BURN_TIME)

    def _lazy_init_window(self):
        if self.window is None:
            self.window = Window("Overcooked V3")

    def show(self, block=False):
        self._lazy_init_window()
        self.window.show(block=block)

    def render(self, state, agent_view_size=None):
        """Render the state in a window."""
        self._lazy_init_window()
        img = self._render_state(state, agent_view_size)
        self.window.show_img(img)
        return img

    def render_state(self, state, agent_view_size=None):
        """Render state to an image array without displaying."""
        return self._render_state(state, agent_view_size)

    def animate(self, state_seq, filename="animation.gif", agent_view_size=None):
        """Animate a gif from a state sequence and save to file."""
        if not HAS_IMAGEIO:
            raise ImportError("imageio is required for animation. Install with: pip install imageio")
        frame_seq = jax.vmap(self._render_state, in_axes=(0, None))(
            state_seq, agent_view_size
        )
        imageio.mimsave(filename, frame_seq, "GIF", duration=0.5)

    def render_sequence(self, state_seq, agent_view_size=None):
        """Render a sequence of states to images."""
        frame_seq = jax.vmap(self._render_state, in_axes=(0, None))(
            state_seq, agent_view_size
        )
        return frame_seq

    @classmethod
    def _encode_agent_extras(cls, direction, idx):
        return direction | (idx << 2)

    @classmethod
    def _decode_agent_extras(cls, extras):
        direction = extras & 0x3
        idx = extras >> 2
        return direction, idx

    @partial(jax.jit, static_argnums=(0, 2))
    def _render_state(self, state, agent_view_size=None):
        """Render the state to an image."""
        grid = state.grid
        agents = state.agents
        recipe = state.recipe
        pot_timers = state.pot_cooking_timer
        pot_positions = state.pot_positions
        pot_active_mask = state.pot_active_mask

        num_agents = agents.dir.shape[0]

        # Include agents in grid for rendering
        def _include_agents(grid, x):
            agent, idx = x
            pos = agent.pos
            inventory = agent.inventory
            direction = agent.dir

            extra_info = OvercookedV3Visualizer._encode_agent_extras(direction, idx)

            new_grid = grid.at[pos.y, pos.x].set(
                [StaticObject.AGENT, inventory, extra_info]
            )
            return new_grid, None

        grid, _ = jax.lax.scan(_include_agents, grid, (agents, jnp.arange(num_agents)))

        # Update pot timers in grid for rendering
        def _update_pot_timer_in_grid(grid, pot_idx):
            pot_y, pot_x = pot_positions[pot_idx]
            timer = pot_timers[pot_idx]
            is_active = pot_active_mask[pot_idx]

            new_grid = jax.lax.select(
                is_active,
                grid.at[pot_y, pot_x, 2].set(timer),
                grid
            )
            return new_grid, None

        grid, _ = jax.lax.scan(
            _update_pot_timer_in_grid, grid, jnp.arange(pot_positions.shape[0])
        )

        static_objects = grid[:, :, 0]

        # Show recipe on recipe indicators
        recipe_indicator_mask = static_objects == StaticObject.RECIPE_INDICATOR
        new_ingredients_layer = jnp.where(
            recipe_indicator_mask,
            recipe | DynamicObject.COOKED | DynamicObject.PLATE,
            grid[:, :, 1],
        )
        grid = grid.at[:, :, 1].set(new_ingredients_layer)

        highlight_mask = jnp.zeros(grid.shape[:2], dtype=bool)

        img = self._render_grid(grid, highlight_mask)
        return img

    @staticmethod
    def _get_ingredient_idx_list(ingredients):
        """Get list of ingredient indices from encoded ingredients."""
        # Strip plate and cooked flags
        ing_only = ingredients >> 2

        # Check each ingredient slot (up to 3 ingredient types)
        idx0 = jax.lax.select((ing_only & 0x3) > 0, 0, -1)
        idx1 = jax.lax.select(((ing_only >> 2) & 0x3) > 0, 1, -1)
        idx2 = jax.lax.select(((ing_only >> 4) & 0x3) > 0, 2, -1)

        # Get counts for each
        count0 = ing_only & 0x3
        count1 = (ing_only >> 2) & 0x3
        count2 = (ing_only >> 4) & 0x3

        # Build result: expand counts into positions
        result = jnp.array([-1, -1, -1])

        # Simple expansion for up to 3 total ingredients
        def add_ingredient(carry, _):
            result, pos, idx0, idx1, idx2, count0, count1, count2 = carry

            # Add from idx0 if count > 0
            use_idx0 = count0 > 0
            result = jax.lax.select(use_idx0, result.at[pos].set(idx0), result)
            pos = jax.lax.select(use_idx0, pos + 1, pos)
            count0 = jax.lax.select(use_idx0, count0 - 1, count0)

            return (result, pos, idx0, idx1, idx2, count0, count1, count2), None

        # Run 3 times to fill up to 3 slots
        (result, _, _, _, _, _, _, _), _ = jax.lax.scan(
            add_ingredient,
            (result, 0, idx0, idx1, idx2, count0, count1, count2),
            None,
            length=3
        )

        return result

    @staticmethod
    def _render_dynamic_item(
        ingredients,
        img,
        plate_fn=rendering.point_in_circle(0.5, 0.5, 0.3),
        ingredient_fn=rendering.point_in_circle(0.5, 0.5, 0.15),
        dish_positions=jnp.array([(0.5, 0.4), (0.4, 0.6), (0.6, 0.6)]),
    ):
        def _no_op(img, ingredients):
            return img

        def _render_plate(img, ingredients):
            return rendering.fill_coords(img, plate_fn, COLORS["white"])

        def _render_ingredient(img, ingredients):
            idx = DynamicObject.get_ingredient_type(ingredients)
            idx = jnp.clip(idx, 0, len(INGREDIENT_COLORS) - 1)
            return rendering.fill_coords(img, ingredient_fn, INGREDIENT_COLORS[idx])

        def _render_dish(img, ingredients):
            img = rendering.fill_coords(img, plate_fn, COLORS["white"])
            ingredient_indices = OvercookedV3Visualizer._get_ingredient_idx_list(ingredients)

            for idx_pos, ingredient_idx in enumerate(ingredient_indices):
                color = INGREDIENT_COLORS[jnp.clip(ingredient_idx, 0, len(INGREDIENT_COLORS) - 1)]
                pos = dish_positions[idx_pos]
                ing_fn = rendering.point_in_circle(pos[0], pos[1], 0.1)
                img_ing = rendering.fill_coords(img, ing_fn, color)
                img = jax.lax.select(ingredient_idx != -1, img_ing, img)

            return img

        branches = jnp.array(
            [
                ingredients == 0,
                ingredients == DynamicObject.PLATE,
                DynamicObject.is_ingredient(ingredients),
                (ingredients & DynamicObject.COOKED) != 0,
            ]
        )
        branch_idx = jnp.argmax(branches)

        img = jax.lax.switch(
            branch_idx,
            [_no_op, _render_plate, _render_ingredient, _render_dish],
            img,
            ingredients,
        )

        return img

    @staticmethod
    def _render_cell(cell, img, pot_cook_time=POT_COOK_TIME, pot_burn_time=POT_BURN_TIME):
        static_object = cell[0]

        def _render_empty(cell, img):
            return img

        def _render_wall(cell, img):
            img = rendering.fill_coords(
                img, rendering.point_in_rect(0, 1, 0, 1), COLORS["grey"]
            )
            img = OvercookedV3Visualizer._render_dynamic_item(cell[1], img)
            return img

        def _render_agent(cell, img):
            tri_fn = rendering.point_in_triangle(
                (0.12, 0.19),
                (0.87, 0.50),
                (0.12, 0.81),
            )

            direction, idx = OvercookedV3Visualizer._decode_agent_extras(cell[2])
            direction_reordering = jnp.array([3, 1, 0, 2])
            direction = direction_reordering[direction]

            agent_color = AGENT_COLORS[jnp.clip(idx, 0, len(AGENT_COLORS) - 1)]

            tri_fn = rendering.rotate_fn(
                tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * direction
            )
            img = rendering.fill_coords(img, tri_fn, agent_color)

            img = OvercookedV3Visualizer._render_dynamic_item(
                cell[1],
                img,
                plate_fn=rendering.point_in_circle(0.75, 0.75, 0.2),
                ingredient_fn=rendering.point_in_circle(0.75, 0.75, 0.15),
                dish_positions=jnp.array([(0.65, 0.65), (0.85, 0.65), (0.75, 0.85)]),
            )

            return img

        def _render_goal(cell, img):
            img = rendering.fill_coords(
                img, rendering.point_in_rect(0, 1, 0, 1), COLORS["grey"]
            )
            img = rendering.fill_coords(
                img, rendering.point_in_rect(0.1, 0.9, 0.1, 0.9), COLORS["green"]
            )
            return img

        def _render_pot(cell, img):
            return OvercookedV3Visualizer._render_pot(cell, img, pot_cook_time, pot_burn_time)

        def _render_recipe_indicator(cell, img):
            img = rendering.fill_coords(
                img, rendering.point_in_rect(0, 1, 0, 1), COLORS["grey"]
            )
            img = rendering.fill_coords(
                img, rendering.point_in_rect(0.1, 0.9, 0.1, 0.9), COLORS["brown"]
            )
            img = OvercookedV3Visualizer._render_dynamic_item(cell[1], img)
            return img

        def _render_plate_pile(cell, img):
            img = rendering.fill_coords(
                img, rendering.point_in_rect(0, 1, 0, 1), COLORS["grey"]
            )
            plate_fns = [
                rendering.point_in_circle(*coord, 0.2)
                for coord in [(0.3, 0.3), (0.75, 0.42), (0.4, 0.75)]
            ]
            for plate_fn in plate_fns:
                img = rendering.fill_coords(img, plate_fn, COLORS["white"])
            return img

        def _render_ingredient_pile(cell, img):
            ingredient_idx = cell[0] - StaticObject.INGREDIENT_PILE_BASE

            img = rendering.fill_coords(
                img, rendering.point_in_rect(0, 1, 0, 1), COLORS["grey"]
            )
            ingredient_fns = [
                rendering.point_in_circle(*coord, 0.15)
                for coord in [
                    (0.5, 0.15),
                    (0.3, 0.4),
                    (0.8, 0.35),
                    (0.4, 0.8),
                    (0.75, 0.75),
                ]
            ]

            for ing_fn in ingredient_fns:
                img = rendering.fill_coords(
                    img, ing_fn, INGREDIENT_COLORS[jnp.clip(ingredient_idx, 0, len(INGREDIENT_COLORS) - 1)]
                )

            return img

        def _render_item_conveyor(cell, img):
            """Render item conveyor belt with direction arrow."""
            img = rendering.fill_coords(
                img, rendering.point_in_rect(0, 1, 0, 1), COLORS["light_grey"]
            )
            # Draw conveyor belt lines
            for i in range(4):
                offset = 0.15 + i * 0.2
                img = rendering.fill_coords(
                    img, rendering.point_in_rect(0.05, 0.95, offset, offset + 0.1), COLORS["grey"]
                )

            # Draw direction arrow
            direction = cell[2] & 0x3
            arrow_fn = rendering.point_in_triangle(
                (0.3, 0.5),
                (0.7, 0.3),
                (0.7, 0.7),
            )
            direction_reordering = jnp.array([1, 3, 2, 0])
            dir_idx = direction_reordering[direction]
            arrow_fn = rendering.rotate_fn(
                arrow_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * dir_idx
            )
            img = rendering.fill_coords(img, arrow_fn, COLORS["orange"])

            # Render any item on the conveyor
            img = OvercookedV3Visualizer._render_dynamic_item(cell[1], img)
            return img

        def _render_player_conveyor(cell, img):
            """Render player conveyor belt with direction arrow."""
            img = rendering.fill_coords(
                img, rendering.point_in_rect(0, 1, 0, 1), COLORS["light_blue"]
            )
            # Draw conveyor belt lines
            for i in range(4):
                offset = 0.15 + i * 0.2
                img = rendering.fill_coords(
                    img, rendering.point_in_rect(0.05, 0.95, offset, offset + 0.1), COLORS["blue"]
                )

            # Draw direction arrow
            direction = cell[2] & 0x3
            arrow_fn = rendering.point_in_triangle(
                (0.3, 0.5),
                (0.7, 0.3),
                (0.7, 0.7),
            )
            direction_reordering = jnp.array([1, 3, 2, 0])
            dir_idx = direction_reordering[direction]
            arrow_fn = rendering.rotate_fn(
                arrow_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * dir_idx
            )
            img = rendering.fill_coords(img, arrow_fn, COLORS["cyan"])

            return img

        # Build render function lookup
        # Map static object types to render functions
        render_fns = [_render_empty] * 25  # Enough for all object types
        render_fns[StaticObject.EMPTY] = _render_empty
        render_fns[StaticObject.WALL] = _render_wall
        render_fns[StaticObject.AGENT] = _render_agent
        render_fns[StaticObject.SELF_AGENT] = _render_empty
        render_fns[StaticObject.GOAL] = _render_goal
        render_fns[StaticObject.POT] = _render_pot
        render_fns[StaticObject.RECIPE_INDICATOR] = _render_recipe_indicator
        render_fns[StaticObject.PLATE_PILE] = _render_plate_pile
        render_fns[StaticObject.ITEM_CONVEYOR] = _render_item_conveyor
        render_fns[StaticObject.PLAYER_CONVEYOR] = _render_player_conveyor

        # Handle ingredient piles (10-19)
        is_ingredient_pile = (static_object >= StaticObject.INGREDIENT_PILE_BASE) & \
                            (static_object < StaticObject.ITEM_CONVEYOR)

        branch_idx = jax.lax.select(
            is_ingredient_pile,
            len(render_fns) - 1,  # Use last slot for ingredient pile
            jnp.clip(static_object, 0, len(render_fns) - 2)
        )

        render_fns[-1] = _render_ingredient_pile

        return jax.lax.switch(
            branch_idx,
            render_fns,
            cell,
            img,
        )

    @staticmethod
    def _render_pot(cell, img, pot_cook_time=POT_COOK_TIME, pot_burn_time=POT_BURN_TIME):
        ingredients = cell[1]
        time_left = cell[2]

        is_cooking = time_left > pot_burn_time
        is_burning = (time_left > 0) & (time_left <= pot_burn_time)
        is_cooked = (ingredients & DynamicObject.COOKED) != 0
        is_burned = (ingredients & DynamicObject.BURNED) != 0
        is_idle = ~is_cooking & ~is_burning & ~is_cooked & ~is_burned

        ingredient_indices = OvercookedV3Visualizer._get_ingredient_idx_list(ingredients)
        has_ingredients = ingredient_indices[0] != -1

        img = rendering.fill_coords(
            img, rendering.point_in_rect(0, 1, 0, 1), COLORS["grey"]
        )

        ingredient_fns = [
            rendering.point_in_circle(*coord, 0.13)
            for coord in [(0.23, 0.33), (0.77, 0.33), (0.50, 0.33)]
        ]

        for i, ingredient_idx in enumerate(ingredient_indices):
            color = INGREDIENT_COLORS[jnp.clip(ingredient_idx, 0, len(INGREDIENT_COLORS) - 1)]
            img_ing = rendering.fill_coords(img, ingredient_fns[i], color)
            img = jax.lax.select(ingredient_idx != -1, img_ing, img)

        pot_fn = rendering.point_in_rect(0.1, 0.9, 0.33, 0.9)
        lid_fn = rendering.point_in_rect(0.1, 0.9, 0.21, 0.25)
        handle_fn = rendering.point_in_rect(0.4, 0.6, 0.16, 0.21)

        lid_fn_open = rendering.rotate_fn(lid_fn, cx=0.1, cy=0.25, theta=-0.1 * math.pi)
        handle_fn_open = rendering.rotate_fn(handle_fn, cx=0.1, cy=0.25, theta=-0.1 * math.pi)
        pot_open = is_idle & has_ingredients

        img = rendering.fill_coords(img, pot_fn, COLORS["black"])

        img_closed = rendering.fill_coords(img, lid_fn, COLORS["black"])
        img_closed = rendering.fill_coords(img_closed, handle_fn, COLORS["black"])

        img_open = rendering.fill_coords(img, lid_fn_open, COLORS["black"])
        img_open = rendering.fill_coords(img_open, handle_fn_open, COLORS["black"])

        img = jax.lax.select(pot_open, img_open, img_closed)

        # Render progress bar (green for cooking, orange for burning window)
        cooking_progress = (pot_cook_time - time_left) / (pot_cook_time - pot_burn_time)
        burning_progress = (pot_burn_time - time_left) / pot_burn_time

        progress_fn_cooking = rendering.point_in_rect(
            0.1, 0.1 + 0.8 * jnp.clip(cooking_progress, 0, 1), 0.83, 0.88
        )
        progress_fn_burning = rendering.point_in_rect(
            0.1, 0.1 + 0.8 * jnp.clip(burning_progress, 0, 1), 0.83, 0.88
        )

        img_cooking = rendering.fill_coords(img, progress_fn_cooking, COLORS["green"])
        img_burning = rendering.fill_coords(img, progress_fn_burning, COLORS["orange"])

        # Show green bar when cooking, orange when in burning window
        img = jax.lax.select(is_cooking, img_cooking, img)
        img = jax.lax.select(is_burning | is_cooked, img_burning, img)

        return img

    def _render_tile(self, obj, highlight=False):
        """Render a single tile."""
        img = jnp.zeros(
            shape=(self.tile_size * self.subdivs, self.tile_size * self.subdivs, 3),
            dtype=jnp.uint8,
        )

        # Draw grid lines
        img = rendering.fill_coords(
            img, rendering.point_in_rect(0, 0.031, 0, 1), COLORS["grey"]
        )
        img = rendering.fill_coords(
            img, rendering.point_in_rect(0, 1, 0, 0.031), COLORS["grey"]
        )

        img = OvercookedV3Visualizer._render_cell(
            obj, img, self.pot_cook_time, self.pot_burn_time
        )

        img_highlight = rendering.highlight_img(img)
        img = jax.lax.select(highlight, img_highlight, img)

        # Downsample for anti-aliasing
        img = rendering.downsample(img, self.subdivs)

        return img

    def _render_grid(self, grid, highlight_mask):
        """Render the full grid."""
        img_grid = jax.vmap(jax.vmap(self._render_tile))(grid, highlight_mask)

        grid_rows, grid_cols, tile_height, tile_width, channels = img_grid.shape

        big_image = img_grid.transpose(0, 2, 1, 3, 4).reshape(
            grid_rows * tile_height, grid_cols * tile_width, channels
        )

        return big_image

    def close(self):
        if self.window is not None:
            self.window.close()
