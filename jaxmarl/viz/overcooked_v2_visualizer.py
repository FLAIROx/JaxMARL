import math
from jaxmarl.environments.overcooked_v2.utils import compute_view_box
from jaxmarl.viz.window import Window
import jaxmarl.viz.grid_rendering_v2 as rendering
import jax
import jax.numpy as jnp
from jaxmarl.environments.overcooked_v2.common import StaticObject, DynamicObject
from jaxmarl.environments.overcooked_v2.settings import (
    POT_COOK_TIME,
    INDICATOR_ACTIVATION_TIME,
)
import imageio
from functools import partial

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
}

INGREDIENT_COLORS = jnp.array(
    [
        COLORS["yellow"],
        COLORS["dark_green"],
        COLORS["purple"],
        COLORS["cyan"],
        COLORS["red"],
        COLORS["orange"],
        COLORS["purple"],
        COLORS["blue"],
        COLORS["pink"],
        COLORS["brown"],
    ]
)


AGENT_COLORS = jnp.array(
    [
        COLORS["red"],
        COLORS["blue"],
        COLORS["green"],
        COLORS["purple"],
        COLORS["yellow"],
        COLORS["orange"],
    ]
)


class OvercookedV2Visualizer:
    """
    Manages a window and renders contents of EnvState instances to it.
    """

    tile_cache = {}

    def __init__(self, tile_size=TILE_PIXELS, subdivs=3):
        self.window = None

        self.tile_size = tile_size
        self.subdivs = subdivs

    def _lazy_init_window(self):
        if self.window is None:
            self.window = Window("Overcooked V2")

    def show(self, block=False):
        self._lazy_init_window()
        self.window.show(block=block)

    def render(self, state, agent_view_size=None):
        """Method for rendering the state in a window. Esp. useful for interactive mode."""
        self._lazy_init_window()

        img = self._render_state(state, agent_view_size)

        self.window.show_img(img)

    def animate(self, state_seq, filename="animation.gif", agent_view_size=None):
        """Animate a gif give a state sequence and save if to file."""

        frame_seq = jax.vmap(self._render_state, in_axes=(0, None))(
            state_seq, agent_view_size
        )
        # print("frame_seq", frame_seq)
        # print("frame_seq.shape", frame_seq.shape)
        # print("frame_seq.dtype", frame_seq.dtype)

        imageio.mimsave(filename, frame_seq, "GIF", duration=0.5)

    def render_sequence(self, state_seq, agent_view_size=None):
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
        """
        Render the state
        """

        grid = state.grid
        agents = state.agents
        recipe = state.recipe

        num_agents = agents.dir.shape[0]

        def _include_agents(grid, x):
            agent, idx = x
            pos = agent.pos
            inventory = agent.inventory
            direction = agent.dir

            # we have to do the encoding because we don't really have a way to also pass the agent's id
            extra_info = OvercookedV2Visualizer._encode_agent_extras(direction, idx)

            new_grid = grid.at[pos.y, pos.x].set(
                [StaticObject.AGENT, inventory, extra_info]
            )
            return new_grid, None

        grid, _ = jax.lax.scan(_include_agents, grid, (agents, jnp.arange(num_agents)))

        static_objects = grid[:, :, 0]
        ingredients = grid[:, :, 1]
        extra_info = grid[:, :, 2]

        recipe_indicator_mask = static_objects == StaticObject.RECIPE_INDICATOR
        button_recipe_indicator_mask = (
            static_objects == StaticObject.BUTTON_RECIPE_INDICATOR
        ) & (extra_info > 0)

        new_ingredients_layer = jnp.where(
            recipe_indicator_mask | button_recipe_indicator_mask,
            recipe | DynamicObject.COOKED | DynamicObject.PLATE,
            ingredients,
        )
        grid = grid.at[:, :, 1].set(new_ingredients_layer)

        highlight_mask = jnp.zeros(grid.shape[:2], dtype=bool)
        if agent_view_size:
            for x, y in zip(agents.pos.x, agents.pos.y):
                x_low, x_high, y_low, y_high = compute_view_box(
                    x, y, agent_view_size, grid.shape[0], grid.shape[1]
                )

                row_mask = jnp.arange(grid.shape[0])
                col_mask = jnp.arange(grid.shape[1])

                row_mask = (row_mask >= y_low) & (row_mask < y_high)
                col_mask = (col_mask >= x_low) & (col_mask < x_high)

                agent_mask = row_mask[:, None] & col_mask[None, :]

                highlight_mask |= agent_mask

        # Render the whole grid
        img = self._render_grid(grid, highlight_mask)
        return img

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
            idx = DynamicObject.get_ingredient_idx(ingredients)
            return rendering.fill_coords(img, ingredient_fn, INGREDIENT_COLORS[idx])

        def _render_dish(img, ingredients):
            img = rendering.fill_coords(img, plate_fn, COLORS["white"])
            ingredient_indices = DynamicObject.get_ingredient_idx_list_jit(ingredients)

            for idx, ingredient_idx in enumerate(ingredient_indices):
                color = INGREDIENT_COLORS[ingredient_idx]
                pos = dish_positions[idx]
                ingredient_fn = rendering.point_in_circle(pos[0], pos[1], 0.1)
                img_ing = rendering.fill_coords(img, ingredient_fn, color)

                img = jax.lax.select(ingredient_idx != -1, img_ing, img)

            return img

        branches = jnp.array(
            [
                ingredients == 0,
                ingredients == DynamicObject.PLATE,
                DynamicObject.is_ingredient(ingredients),
                ingredients & DynamicObject.COOKED,
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
    def _render_cell(cell, img):
        static_object = cell[0]

        def _render_empty(cell, img):
            return img

        def _render_wall(cell, img):
            img = rendering.fill_coords(
                img, rendering.point_in_rect(0, 1, 0, 1), COLORS["grey"]
            )
            img = OvercookedV2Visualizer._render_dynamic_item(cell[1], img)

            return img

        def _render_agent(cell, img):
            tri_fn = rendering.point_in_triangle(
                (0.12, 0.19),
                (0.87, 0.50),
                (0.12, 0.81),
            )

            direction, idx = OvercookedV2Visualizer._decode_agent_extras(cell[2])

            # A bit hacky, but needed so that actions order matches the one of Overcooked-AI
            direction_reordering = jnp.array([3, 1, 0, 2])
            direction = direction_reordering[direction]

            agent_color = AGENT_COLORS[idx]

            tri_fn = rendering.rotate_fn(
                tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * direction
            )
            img = rendering.fill_coords(img, tri_fn, agent_color)

            img = OvercookedV2Visualizer._render_dynamic_item(
                cell[1],
                img,
                plate_fn=rendering.point_in_circle(0.75, 0.75, 0.2),
                ingredient_fn=rendering.point_in_circle(0.75, 0.75, 0.15),
                dish_positions=jnp.array([(0.65, 0.65), (0.85, 0.65), (0.75, 0.85)]),
            )

            return img

        def _render_agent_self(cell, img):
            # Note: This should not ever be called
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
            return OvercookedV2Visualizer._render_pot(cell, img)

        def _render_recipe_indicator(cell, img):
            img = rendering.fill_coords(
                img, rendering.point_in_rect(0, 1, 0, 1), COLORS["grey"]
            )
            img = rendering.fill_coords(
                img, rendering.point_in_rect(0.1, 0.9, 0.1, 0.9), COLORS["brown"]
            )
            img = OvercookedV2Visualizer._render_dynamic_item(cell[1], img)

            return img

        def _render_button_recipe_indicator(cell, img):
            img = rendering.fill_coords(
                img, rendering.point_in_rect(0, 1, 0, 1), COLORS["grey"]
            )
            img = rendering.fill_coords(
                img, rendering.point_in_rect(0.1, 0.9, 0.1, 0.9), COLORS["brown"]
            )
            img = OvercookedV2Visualizer._render_dynamic_item(cell[1], img)

            time_left = cell[2]
            progress_fn = rendering.point_in_rect(
                0.1,
                0.9 - (0.9 - 0.1) / INDICATOR_ACTIVATION_TIME * time_left,
                0.83,
                0.88,
            )
            img_timer = rendering.fill_coords(img, progress_fn, COLORS["green"])

            button_fn = rendering.point_in_circle(0.5, 0.5, 0.2)
            img_button = rendering.fill_coords(img, button_fn, COLORS["red"])

            img = jax.lax.select(time_left > 0, img_timer, img_button)
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

            for ingredient_fn in ingredient_fns:
                img = rendering.fill_coords(
                    img, ingredient_fn, INGREDIENT_COLORS[ingredient_idx]
                )

            return img

        render_fns_dict = {
            StaticObject.EMPTY: _render_empty,
            StaticObject.WALL: _render_wall,
            StaticObject.AGENT: _render_agent,
            StaticObject.SELF_AGENT: _render_agent_self,
            StaticObject.GOAL: _render_goal,
            StaticObject.POT: _render_pot,
            StaticObject.RECIPE_INDICATOR: _render_recipe_indicator,
            StaticObject.BUTTON_RECIPE_INDICATOR: _render_button_recipe_indicator,
            StaticObject.PLATE_PILE: _render_plate_pile,
        }

        render_fns = [_render_empty] * (max(render_fns_dict.keys()) + 2)
        for key, value in render_fns_dict.items():
            render_fns[key] = value
        render_fns[-1] = _render_ingredient_pile

        branch_idx = jnp.clip(static_object, 0, len(render_fns) - 1)

        return jax.lax.switch(
            branch_idx,
            render_fns,
            cell,
            img,
        )

    @staticmethod
    def _render_pot(cell, img):
        ingredients = cell[1]
        time_left = cell[2]

        is_cooking = time_left > 0
        is_cooked = (ingredients & DynamicObject.COOKED) != 0
        is_idle = ~is_cooking & ~is_cooked
        ingredients = DynamicObject.get_ingredient_idx_list_jit(ingredients)
        has_ingredients = ingredients[0] != -1

        img = rendering.fill_coords(
            img, rendering.point_in_rect(0, 1, 0, 1), COLORS["grey"]
        )

        ingredient_fns = [
            rendering.point_in_circle(*coord, 0.13)
            for coord in [(0.23, 0.33), (0.77, 0.33), (0.50, 0.33)]
        ]

        for i, ingredient_idx in enumerate(ingredients):
            img_ing = rendering.fill_coords(
                img, ingredient_fns[i], INGREDIENT_COLORS[ingredient_idx]
            )
            img = jax.lax.select(ingredient_idx != -1, img_ing, img)

        pot_fn = rendering.point_in_rect(0.1, 0.9, 0.33, 0.9)
        lid_fn = rendering.point_in_rect(0.1, 0.9, 0.21, 0.25)
        handle_fn = rendering.point_in_rect(0.4, 0.6, 0.16, 0.21)

        lid_fn_open = rendering.rotate_fn(lid_fn, cx=0.1, cy=0.25, theta=-0.1 * math.pi)
        handle_fn_open = rendering.rotate_fn(
            handle_fn, cx=0.1, cy=0.25, theta=-0.1 * math.pi
        )
        pot_open = is_idle & has_ingredients

        img = rendering.fill_coords(img, pot_fn, COLORS["black"])

        img_closed = rendering.fill_coords(img, lid_fn, COLORS["black"])
        img_closed = rendering.fill_coords(img_closed, handle_fn, COLORS["black"])

        img_open = rendering.fill_coords(img, lid_fn_open, COLORS["black"])
        img_open = rendering.fill_coords(img_open, handle_fn_open, COLORS["black"])

        img = jax.lax.select(pot_open, img_open, img_closed)

        # Render progress bar
        progress_fn = rendering.point_in_rect(
            0.1, 0.9 - (0.9 - 0.1) / POT_COOK_TIME * time_left, 0.83, 0.88
        )
        img_timer = rendering.fill_coords(img, progress_fn, COLORS["green"])
        img = jax.lax.select(is_cooking, img_timer, img)

        return img

    def _render_tile(
        self,
        obj,
        highlight=False,
    ):
        """
        Render a tile and cache the result
        """
        # key = (*obj.tolist(), highlight, tile_size)

        # if key in OvercookedV2Visualizer.tile_cache:
        #     return OvercookedV2Visualizer.tile_cache[key]

        img = jnp.zeros(
            shape=(self.tile_size * self.subdivs, self.tile_size * self.subdivs, 3),
            dtype=jnp.uint8,
        )

        # Draw the grid lines (top and left edges)
        img = rendering.fill_coords(
            img, rendering.point_in_rect(0, 0.031, 0, 1), COLORS["grey"]
        )
        img = rendering.fill_coords(
            img, rendering.point_in_rect(0, 1, 0, 0.031), COLORS["grey"]
        )

        img = OvercookedV2Visualizer._render_cell(obj, img)

        img_highlight = rendering.highlight_img(img)
        img = jax.lax.select(highlight, img_highlight, img)

        # Downsample the image to perform supersampling/anti-aliasing
        img = rendering.downsample(img, self.subdivs)

        # Cache the rendered tile
        # OvercookedV2Visualizer.tile_cache[key] = img

        return img

    def _render_grid(
        self,
        grid,
        highlight_mask,
    ):
        img_grid = jax.vmap(jax.vmap(self._render_tile))(grid, highlight_mask)

        # print("img_grid", img_grid.shape)

        grid_rows, grid_cols, tile_height, tile_width, channels = img_grid.shape

        big_image = img_grid.transpose(0, 2, 1, 3, 4).reshape(
            grid_rows * tile_height, grid_cols * tile_width, channels
        )

        # print("big_image", big_image.shape)

        return big_image

    def close(self):
        self.window.close()
