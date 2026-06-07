"""Overcooked V3 Environment with pot burning, order queue, and conveyor belts."""

from enum import Enum
from typing import List, Optional, Union, Tuple, Dict
import warnings
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
import chex

from jaxmarl.environments import MultiAgentEnv
from jaxmarl.environments import spaces
from jaxmarl.environments.overcooked_v3.common import (
    ACTION_TO_DIRECTION,
    DIR_TO_VEC,
    MAX_INGREDIENTS,
    Actions,
    StaticObject,
    DynamicObject,
    Direction,
    ButtonAction,
    Position,
    Agent,
)
from jaxmarl.environments.overcooked_v3.layouts import overcooked_v3_layouts, Layout
from jaxmarl.environments.overcooked_v3.settings import (
    DELIVERY_REWARD,
    POT_COOK_TIME,
    POT_BURN_TIME,
    ORDER_EXPIRED_PENALTY,
    DEFAULT_ORDER_GENERATION_RATE,
    DEFAULT_ORDER_EXPIRATION_TIME,
    DEFAULT_MAX_ORDERS,
    SHAPED_REWARDS,
    MAX_POTS,
    MAX_ITEM_CONVEYORS,
    MAX_PLAYER_CONVEYORS,
    MAX_MOVING_WALLS,
    MAX_BUTTONS,
    MAX_BUTTON_TARGETS,
    MAX_BARRIERS,
    DEFAULT_BARRIER_DURATION,
)
from jaxmarl.environments.overcooked_v3.utils import (
    tree_select,
    compute_enclosed_spaces,
)


# =============================================================================
# Observation Types and State Container
# =============================================================================


class ObservationType(str, Enum):
    DEFAULT = "default"
    FEATURIZED = "featurized"


@chex.dataclass
class State:
    """Environment state for Overcooked V3."""

    agents: Agent

    # Grid: height x width x 3 channels
    # Channel 0: static objects
    # Channel 1: dynamic items (plates, ingredients, soups)
    # Channel 2: extra info (pot timers, conveyor directions)
    grid: chex.Array

    # Pot state (fixed size arrays for JIT compatibility)
    # pot_positions stores (y, x) for each pot, pot_active_mask indicates valid pots
    pot_positions: chex.Array  # [max_pots, 2] - (y, x) positions
    pot_cooking_timer: (
        chex.Array
    )  # [max_pots] - countdown to cooked (0 when idle/cooked)
    pot_active_mask: chex.Array  # [max_pots] - bool, which pot slots are valid

    # Order queue state (optional feature)
    order_types: chex.Array  # [max_orders] - SoupType enum values
    order_expirations: chex.Array  # [max_orders] - steps remaining
    order_active_mask: chex.Array  # [max_orders] - bool, which order slots are valid

    # Item conveyor state
    item_conveyor_positions: chex.Array  # [max_item_conveyors, 2] - (y, x)
    item_conveyor_directions: chex.Array  # [max_item_conveyors] - Direction enum
    item_conveyor_active_mask: chex.Array  # [max_item_conveyors] - bool

    # Player conveyor state
    player_conveyor_positions: chex.Array  # [max_player_conveyors, 2] - (y, x)
    player_conveyor_directions: chex.Array  # [max_player_conveyors] - Direction enum
    player_conveyor_active_mask: chex.Array  # [max_player_conveyors] - bool

    # Moving wall state
    moving_wall_positions: chex.Array  # [max_moving_walls, 2] - (y, x)
    moving_wall_directions: chex.Array  # [max_moving_walls] - Direction enum
    moving_wall_active_mask: chex.Array  # [max_moving_walls] - bool
    moving_wall_paused: chex.Array  # [max_moving_walls] - bool
    moving_wall_bounce: chex.Array  # [max_moving_walls] - bool

    # Button state
    button_positions: chex.Array  # [max_buttons, 2] - (y, x)
    button_target_idxs: chex.Array  # [max_buttons, max_button_targets]
    button_target_mask: chex.Array  # [max_buttons, max_button_targets] - bool
    button_action_type: chex.Array  # [max_buttons] - ButtonAction enum
    button_active_mask: chex.Array  # [max_buttons] - bool
    button_toggled: chex.Array  # [max_buttons] - bool (current toggle state)

    # Barrier state
    barrier_positions: chex.Array  # [max_barriers, 2] - (y, x)
    barrier_active: chex.Array  # [max_barriers] - bool (blocks when True)
    barrier_active_mask: chex.Array  # [max_barriers] - bool (which slots are valid)
    barrier_timer: (
        chex.Array
    )  # [max_barriers] - steps until reactivation (0 means permanent state)
    barrier_duration: (
        chex.Array
    )  # [max_barriers] - configured duration for timed deactivation

    # Episode state
    time: chex.Array
    terminal: bool
    recipe: int  # Current target recipe (bit-encoded)

    # Delivery tracking
    new_correct_delivery: bool = False


# =============================================================================
# Overcooked V3 Environment
# =============================================================================


class OvercookedV3(MultiAgentEnv):
    """Overcooked V3 environment with pot burning, order queue, and conveyors.

    Methods:
        reset(key) -> Tuple[Dict[str, Array], State]:
            Reset the environment and return initial observations and state.

        step_env(key, state, actions) -> Tuple[obs, State, rewards, dones, info]:
            Perform a single timestep: process actions, conveyors, orders, and check termination.

        step_agents(key, state, actions) -> Tuple[State, float, Array]:
            Process agent movement (with collision resolution) and interact actions.

        process_interact(grid, agent, all_inventories, recipe, pot_timers,
            pot_positions, pot_active_mask) -> Tuple[grid, agent, correct_delivery,
            reward, shaped_reward, pot_timers]:
            Handle a single agent's interact action (pickup, drop, cook, deliver).

        is_terminal(state) -> bool:
            Check whether the episode is done (max steps reached).

        get_obs(state) -> Dict[str, Array]:
            Get observations for all agents, dispatching by per-agent observation type.

        get_obs_for_type(state, obs_type) -> Dict[str, Array]:
            Get observations for a specific observation type (default or featurized).

        get_obs_default(state) -> Array:
            Build default grid-based observation tensors for all agents.

        name (property) -> str:
            Return the environment name.

        num_actions (property) -> int:
            Return the number of possible actions.

        action_space(agent_id) -> spaces.Discrete:
            Return the discrete action space.

        observation_space(agent_id) -> spaces.Box:
            Return the box observation space.
    """

    # -------------------------------------------------------------------------
    # Construction and Layout Preprocessing
    # -------------------------------------------------------------------------

    def __init__(
        self,
        layout: Union[str, Layout] = "cramped_room",
        max_steps: int = 400,
        observation_type: Union[
            ObservationType, List[ObservationType]
        ] = ObservationType.DEFAULT,
        agent_view_size: Optional[int] = None,
        # Pot settings
        pot_cook_time: int = POT_COOK_TIME,
        pot_burn_time: int = POT_BURN_TIME,
        # Order queue settings
        enable_order_queue: bool = False,
        max_orders: int = DEFAULT_MAX_ORDERS,
        order_generation_rate: float = DEFAULT_ORDER_GENERATION_RATE,
        order_expiration_time: int = DEFAULT_ORDER_EXPIRATION_TIME,
        # Conveyor belt settings
        enable_item_conveyors: Optional[bool] = None,
        enable_player_conveyors: Optional[bool] = None,
        # Moving wall and button settings
        enable_moving_walls: Optional[bool] = None,
        enable_buttons: Optional[bool] = None,
        # Barrier settings
        barrier_duration: Union[int, List[int]] = DEFAULT_BARRIER_DURATION,
        # Reward settings
        delivery_reward: float = DELIVERY_REWARD,
        shaped_rewards: bool = True,
        # Random initialization
        random_reset: bool = False,
        random_agent_positions: bool = False,
    ):
        """Initialize the Overcooked V3 environment.

        Args:
            layout: Layout name or Layout object
            max_steps: Maximum steps per episode
            observation_type: Type of observation (default or featurized)
            agent_view_size: Partial observability window size (None for full)
            pot_cook_time: Steps to cook a full pot (default 90)
            pot_burn_time: Steps in burning window before pot burns (default 60)
            enable_order_queue: Whether to use order queue system
            max_orders: Maximum orders in queue
            order_generation_rate: Probability of new order each step
            order_expiration_time: Steps before order expires
            enable_item_conveyors: Whether item conveyors move items. If None,
                inferred from whether the layout contains item conveyors.
            enable_player_conveyors: Whether player conveyors push agents. If
                None, inferred from whether the layout contains player conveyors.
            enable_moving_walls: Whether moving walls move each step. If None,
                inferred from whether the layout contains moving walls.
            enable_buttons: Whether buttons can be interacted with. If None,
                inferred from whether the layout contains buttons.
            barrier_duration: Duration (steps) for timed barrier deactivation.
                Can be int (same for all) or list of ints per barrier.
            delivery_reward: Reward for correct delivery
            shaped_rewards: Whether to use shaped intermediate rewards
            random_reset: Randomize state on reset
            random_agent_positions: Randomize agent positions only
        """
        if isinstance(layout, str):
            if layout not in overcooked_v3_layouts:
                raise ValueError(
                    f"Invalid layout: {layout}, allowed layouts: {list(overcooked_v3_layouts.keys())}"
                )
            layout = overcooked_v3_layouts[layout]
        elif not isinstance(layout, Layout):
            raise ValueError("Invalid layout, must be a Layout object or a string key")

        is_playable, validation_messages = layout.validate_playable()
        if not is_playable:
            formatted_messages = "\n".join(
                f"- {message}" for message in validation_messages
            )
            raise ValueError(f"Invalid OvercookedV3 layout:\n{formatted_messages}")

        num_agents = len(layout.agent_positions)
        super().__init__(num_agents=num_agents)

        self.height = layout.height
        self.width = layout.width
        self.layout = layout

        self.agents = [f"agent_{i}" for i in range(num_agents)]
        self.action_set = jnp.array(list(Actions))

        if isinstance(observation_type, list):
            if len(observation_type) != num_agents:
                raise ValueError(
                    "Number of observation types must match number of agents"
                )
        self.observation_type = observation_type
        self.agent_view_size = agent_view_size

        self.max_steps = max_steps

        # Pot settings
        self.pot_cook_time = pot_cook_time
        self.pot_burn_time = pot_burn_time

        # Order queue settings
        self.enable_order_queue = enable_order_queue
        self.max_orders = max_orders
        self.order_generation_rate = order_generation_rate
        self.order_expiration_time = order_expiration_time

        # Conveyor settings
        layout_has_item_conveyors = len(layout.item_conveyor_info) > 0
        if enable_item_conveyors is None:
            self.enable_item_conveyors = layout_has_item_conveyors
        else:
            self.enable_item_conveyors = enable_item_conveyors
            if layout_has_item_conveyors and not enable_item_conveyors:
                warnings.warn(
                    "Layout contains item conveyors, but "
                    "enable_item_conveyors=False. Item conveyors will be inert.",
                    UserWarning,
                    stacklevel=2,
                )

        layout_has_player_conveyors = len(layout.player_conveyor_info) > 0
        if enable_player_conveyors is None:
            self.enable_player_conveyors = layout_has_player_conveyors
        else:
            self.enable_player_conveyors = enable_player_conveyors
            if layout_has_player_conveyors and not enable_player_conveyors:
                warnings.warn(
                    "Layout contains player conveyors, but "
                    "enable_player_conveyors=False. Player conveyors will be inert.",
                    UserWarning,
                    stacklevel=2,
                )

        # Moving wall and button settings
        layout_has_moving_walls = len(layout.moving_wall_info) > 0
        if enable_moving_walls is None:
            self.enable_moving_walls = layout_has_moving_walls
        else:
            self.enable_moving_walls = enable_moving_walls
            if layout_has_moving_walls and not enable_moving_walls:
                warnings.warn(
                    "Layout contains moving walls, but "
                    "enable_moving_walls=False. Moving walls will be inert.",
                    UserWarning,
                    stacklevel=2,
                )

        layout_has_buttons = len(layout.button_info) > 0
        if enable_buttons is None:
            self.enable_buttons = layout_has_buttons
        else:
            self.enable_buttons = enable_buttons
            if layout_has_buttons and not enable_buttons:
                warnings.warn(
                    "Layout contains buttons, but enable_buttons=False. "
                    "Buttons will be inert.",
                    UserWarning,
                    stacklevel=2,
                )

        # Barrier settings
        self.barrier_duration = barrier_duration

        # Reward settings
        self.delivery_reward = delivery_reward
        self.shaped_rewards_enabled = shaped_rewards

        # Random reset
        self.random_reset = random_reset
        self.random_agent_positions = random_agent_positions

        # Pre-compute possible recipes
        self.possible_recipes = jnp.array(layout.possible_recipes, dtype=jnp.int32)

        # Pre-compute enclosed spaces for random agent placement
        self.enclosed_spaces = compute_enclosed_spaces(
            layout.static_objects == StaticObject.EMPTY,
        )

        # Compute observation shape
        self.obs_shape = self._get_obs_shape()

        # Extract pot positions from layout
        pot_mask = layout.static_objects == StaticObject.POT
        pot_indices = np.argwhere(pot_mask)
        self.num_pots = min(len(pot_indices), MAX_POTS)
        self._pot_positions = np.zeros((MAX_POTS, 2), dtype=np.int32)
        self._pot_active_mask = np.zeros(MAX_POTS, dtype=bool)
        for i, (y, x) in enumerate(pot_indices[:MAX_POTS]):
            self._pot_positions[i] = [y, x]
            self._pot_active_mask[i] = True

        # Extract conveyor info from layout
        self._item_conveyor_positions = np.zeros(
            (MAX_ITEM_CONVEYORS, 2), dtype=np.int32
        )
        self._item_conveyor_directions = np.zeros(MAX_ITEM_CONVEYORS, dtype=np.int32)
        self._item_conveyor_active_mask = np.zeros(MAX_ITEM_CONVEYORS, dtype=bool)
        for i, (y, x, direction) in enumerate(
            layout.item_conveyor_info[:MAX_ITEM_CONVEYORS]
        ):
            self._item_conveyor_positions[i] = [y, x]
            self._item_conveyor_directions[i] = direction
            self._item_conveyor_active_mask[i] = True

        self._player_conveyor_positions = np.zeros(
            (MAX_PLAYER_CONVEYORS, 2), dtype=np.int32
        )
        self._player_conveyor_directions = np.zeros(
            MAX_PLAYER_CONVEYORS, dtype=np.int32
        )
        self._player_conveyor_active_mask = np.zeros(MAX_PLAYER_CONVEYORS, dtype=bool)
        for i, (y, x, direction) in enumerate(
            layout.player_conveyor_info[:MAX_PLAYER_CONVEYORS]
        ):
            self._player_conveyor_positions[i] = [y, x]
            self._player_conveyor_directions[i] = direction
            self._player_conveyor_active_mask[i] = True

        # Extract moving wall info from layout
        self._moving_wall_positions = np.zeros((MAX_MOVING_WALLS, 2), dtype=np.int32)
        self._moving_wall_directions = np.zeros(MAX_MOVING_WALLS, dtype=np.int32)
        self._moving_wall_active_mask = np.zeros(MAX_MOVING_WALLS, dtype=bool)
        self._moving_wall_bounce = np.zeros(MAX_MOVING_WALLS, dtype=bool)
        for i, (y, x, direction, bounce) in enumerate(
            layout.moving_wall_info[:MAX_MOVING_WALLS]
        ):
            self._moving_wall_positions[i] = [y, x]
            self._moving_wall_directions[i] = direction
            self._moving_wall_active_mask[i] = True
            self._moving_wall_bounce[i] = bounce

        # Extract button info from layout
        self._button_positions = np.zeros((MAX_BUTTONS, 2), dtype=np.int32)
        self._button_target_idxs = np.zeros(
            (MAX_BUTTONS, MAX_BUTTON_TARGETS), dtype=np.int32
        )
        self._button_target_mask = np.zeros(
            (MAX_BUTTONS, MAX_BUTTON_TARGETS), dtype=bool
        )
        self._button_action_type = np.zeros(MAX_BUTTONS, dtype=np.int32)
        self._button_active_mask = np.zeros(MAX_BUTTONS, dtype=bool)
        for i, (y, x, target_idxs, action_type) in enumerate(
            layout.button_info[:MAX_BUTTONS]
        ):
            self._button_positions[i] = [y, x]
            if isinstance(target_idxs, list):
                target_idxs = tuple(target_idxs)
            elif not isinstance(target_idxs, tuple):
                target_idxs = (target_idxs,)
            for j, target_idx in enumerate(target_idxs[:MAX_BUTTON_TARGETS]):
                self._button_target_idxs[i, j] = target_idx
                self._button_target_mask[i, j] = True
            self._button_action_type[i] = action_type
            self._button_active_mask[i] = True

        self._moving_wall_initial_paused = np.zeros(MAX_MOVING_WALLS, dtype=bool)
        for button_idx in range(MAX_BUTTONS):
            if (
                self._button_active_mask[button_idx]
                and self._button_action_type[button_idx] == ButtonAction.TRIGGER_MOVE
            ):
                for target_slot in range(MAX_BUTTON_TARGETS):
                    if self._button_target_mask[button_idx, target_slot]:
                        target_idx = self._button_target_idxs[button_idx, target_slot]
                        self._moving_wall_initial_paused[target_idx] = True

        # Extract barrier info from layout
        self._barrier_positions = np.zeros((MAX_BARRIERS, 2), dtype=np.int32)
        self._barrier_initial_active = np.zeros(MAX_BARRIERS, dtype=bool)
        self._barrier_active_mask = np.zeros(MAX_BARRIERS, dtype=bool)
        self._barrier_duration_config = np.zeros(MAX_BARRIERS, dtype=np.int32)

        num_barriers = len(layout.barrier_info)
        for i, (y, x, active) in enumerate(layout.barrier_info[:MAX_BARRIERS]):
            self._barrier_positions[i] = [y, x]
            self._barrier_initial_active[i] = active
            self._barrier_active_mask[i] = True

            # Set duration for each barrier
            if isinstance(barrier_duration, list):
                if i < len(barrier_duration):
                    self._barrier_duration_config[i] = barrier_duration[i]
                else:
                    self._barrier_duration_config[i] = DEFAULT_BARRIER_DURATION
            else:
                self._barrier_duration_config[i] = barrier_duration

    def _get_obs_shape(self) -> Tuple[int, ...]:
        """Calculate observation shape based on observation type."""
        if self.agent_view_size:
            view_size = self.agent_view_size * 2 + 1
            view_width = min(self.width, view_size)
            view_height = min(self.height, view_size)
        else:
            view_width = self.width
            view_height = self.height

        def _get_obs_shape_single(obs_type):
            if obs_type == ObservationType.DEFAULT:
                num_ingredients = self.layout.num_ingredients
                # Layers breakdown:
                # - agent_layer: 1 (pos) + 4 (dir) + (2 + num_ing) (inv) = 7 + num_ing
                # - other_agent_layers: same = 7 + num_ing
                # - static_layers: 10 (wall, goal, pot, recipe, plate, item_conv, player_conv, moving_wall, button, barrier)
                # - ingredient_pile_layers: num_ing
                # - ingredients_layers: 2 + num_ing
                # - recipe_layers: 2 + num_ing
                # - extra_layers: 1 (pot timer)
                # Total: 29 + 5 * num_ingredients
                num_layers = 29 + 5 * num_ingredients
                return (view_height, view_width, num_layers)
            elif obs_type == ObservationType.FEATURIZED:
                # Simplified feature vector
                return (64,)  # Placeholder
            else:
                raise ValueError(f"Invalid observation type: {obs_type}")

        if isinstance(self.observation_type, list):
            return [
                _get_obs_shape_single(obs_type) for obs_type in self.observation_type
            ]

        return _get_obs_shape_single(self.observation_type)

    # -------------------------------------------------------------------------
    # Reset and Random Initialization
    # -------------------------------------------------------------------------

    def reset(
        self,
        key: chex.PRNGKey,
    ) -> Tuple[Dict[str, chex.Array], State]:
        """Reset the environment."""
        layout = self.layout

        # Initialize grid
        static_objects = layout.static_objects
        grid = jnp.stack(
            [
                static_objects,
                jnp.zeros_like(static_objects),  # dynamic items
                jnp.zeros_like(
                    static_objects
                ),  # extra info (pot timers, conveyor dirs)
            ],
            axis=-1,
            dtype=jnp.int32,
        )

        # Store conveyor directions in extra channel
        for i, (y, x, direction) in enumerate(layout.item_conveyor_info):
            grid = grid.at[y, x, 2].set(direction)
        for i, (y, x, direction) in enumerate(layout.player_conveyor_info):
            grid = grid.at[y, x, 2].set(direction)

        # Store moving wall directions in extra channel
        for y, x, direction, _bounce in layout.moving_wall_info:
            grid = grid.at[y, x, 2].set(direction)

        # Initialize agents
        num_agents = self.num_agents
        x_positions, y_positions = map(jnp.array, zip(*layout.agent_positions))
        agents = Agent(
            pos=Position(x=x_positions, y=y_positions),
            dir=jnp.full((num_agents,), Direction.UP),
            inventory=jnp.zeros((num_agents,), dtype=jnp.int32),
        )

        # Sample recipe
        key, subkey = jax.random.split(key)
        recipe = self._sample_recipe(subkey)

        # Initialize state
        state = State(
            agents=agents,
            grid=grid,
            pot_positions=jnp.array(self._pot_positions),
            pot_cooking_timer=jnp.zeros(MAX_POTS, dtype=jnp.int32),
            pot_active_mask=jnp.array(self._pot_active_mask),
            order_types=jnp.zeros(self.max_orders, dtype=jnp.int32),
            order_expirations=jnp.zeros(self.max_orders, dtype=jnp.int32),
            order_active_mask=jnp.zeros(self.max_orders, dtype=jnp.bool_),
            item_conveyor_positions=jnp.array(self._item_conveyor_positions),
            item_conveyor_directions=jnp.array(self._item_conveyor_directions),
            item_conveyor_active_mask=jnp.array(self._item_conveyor_active_mask),
            player_conveyor_positions=jnp.array(self._player_conveyor_positions),
            player_conveyor_directions=jnp.array(self._player_conveyor_directions),
            player_conveyor_active_mask=jnp.array(self._player_conveyor_active_mask),
            moving_wall_positions=jnp.array(self._moving_wall_positions),
            moving_wall_directions=jnp.array(self._moving_wall_directions),
            moving_wall_active_mask=jnp.array(self._moving_wall_active_mask),
            moving_wall_paused=jnp.array(self._moving_wall_initial_paused),
            moving_wall_bounce=jnp.array(self._moving_wall_bounce),
            button_positions=jnp.array(self._button_positions),
            button_target_idxs=jnp.array(self._button_target_idxs),
            button_target_mask=jnp.array(self._button_target_mask),
            button_action_type=jnp.array(self._button_action_type),
            button_active_mask=jnp.array(self._button_active_mask),
            button_toggled=jnp.zeros(MAX_BUTTONS, dtype=jnp.bool_),
            barrier_positions=jnp.array(self._barrier_positions),
            barrier_active=jnp.array(self._barrier_initial_active),
            barrier_active_mask=jnp.array(self._barrier_active_mask),
            barrier_timer=jnp.zeros(MAX_BARRIERS, dtype=jnp.int32),
            barrier_duration=jnp.array(self._barrier_duration_config),
            time=jnp.array(0),
            terminal=False,
            recipe=recipe,
            new_correct_delivery=False,
        )

        # Optional random initialization
        key, key_randomize = jax.random.split(key)
        if self.random_reset:
            state = self._randomize_state(state, key_randomize)
        elif self.random_agent_positions:
            state = self._randomize_agent_positions(state, key_randomize)

        obs = self.get_obs(state)

        return lax.stop_gradient(obs), lax.stop_gradient(state)

    def _sample_recipe(self, key: chex.PRNGKey) -> int:
        """Sample a recipe from possible recipes."""
        recipe_idx = jax.random.randint(key, (), 0, self.possible_recipes.shape[0])
        recipe = self.possible_recipes[recipe_idx]
        return DynamicObject.get_recipe_encoding(recipe)

    @staticmethod
    def _is_agent_walkable(static_object, pos, state):
        # Check if destination is a barrier and if it's active
        is_barrier_tile = static_object == StaticObject.BARRIER
        barrier_blocks = False
        # Check all barriers to see if any active barrier is at pos

        at_barrier_pos = (
            (state.barrier_positions[:, 0] == pos.y)
            & (state.barrier_positions[:, 1] == pos.x)
            & state.barrier_active_mask[:]
        )

        barrier_blocks = jnp.any(at_barrier_pos & state.barrier_active)

        is_walkable = (
            (static_object == StaticObject.EMPTY)
            | (static_object == StaticObject.PLAYER_CONVEYOR)
            | (
                is_barrier_tile & ~barrier_blocks
            )  # Barrier is walkable if inactive
        )

        return is_walkable

    def _randomize_agent_positions(self, state: State, key: chex.PRNGKey) -> State:
        """Randomize agent positions within their rooms."""
        num_agents = self.num_agents
        agents = state.agents

        def _select_agent_position(taken_mask, x):
            pos, key = x

            allowed_positions = (
                self.enclosed_spaces == self.enclosed_spaces[pos.y, pos.x]
            ) & ~taken_mask
            allowed_positions = allowed_positions.flatten()

            p = allowed_positions / jnp.sum(allowed_positions)
            agent_pos_idx = jax.random.choice(key, allowed_positions.size, (), p=p)
            agent_position = Position(
                x=agent_pos_idx % self.width, y=agent_pos_idx // self.width
            )

            new_taken_mask = taken_mask.at[agent_position.y, agent_position.x].set(True)
            return new_taken_mask, agent_position

        taken_mask = jnp.zeros_like(self.enclosed_spaces, dtype=jnp.bool_)
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, num_agents)
        _, agent_positions = jax.lax.scan(
            _select_agent_position, taken_mask, (agents.pos, keys)
        )

        key, subkey = jax.random.split(key)
        directions = jax.random.randint(subkey, (num_agents,), 0, len(Direction))

        return state.replace(agents=agents.replace(pos=agent_positions, dir=directions))

    def _randomize_state(self, state: State, key: chex.PRNGKey) -> State:
        """Randomize the full state."""
        key, subkey = jax.random.split(key)
        state = self._randomize_agent_positions(state, subkey)
        # Could add more randomization here (pot contents, items on counters, etc.)
        return state

    # -------------------------------------------------------------------------
    # Environment Step Pipeline
    # -------------------------------------------------------------------------

    def step_env(
        self,
        key: chex.PRNGKey,
        state: State,
        actions: Dict[str, chex.Array],
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """Perform a single timestep state transition."""

        acts = self.action_set.take(
            indices=jnp.array([actions[f"agent_{i}"] for i in range(self.num_agents)])
        )

        state, reward, shaped_rewards = self.step_agents(key, state, acts)

        # Process moving walls (before conveyors so conveyors interact with new wall positions)
        if self.enable_moving_walls:
            state = self._process_moving_walls(state)

        # Process conveyors
        if self.enable_item_conveyors:
            state = self._process_item_conveyors(state)
        if self.enable_player_conveyors:
            state = self._process_player_conveyors(state)

        # Process barrier timers (for timed barriers)
        state = self._process_barrier_timers(state)

        # Process order queue
        if self.enable_order_queue:
            key, subkey = jax.random.split(key)
            state, order_reward = self._process_order_queue(state, subkey)
            reward = reward + order_reward

        # Update time
        state = state.replace(time=state.time + 1)

        # Check termination
        done = self.is_terminal(state)
        state = state.replace(terminal=done)

        obs = self.get_obs(state)

        rewards = {f"agent_{i}": reward for i in range(self.num_agents)}
        shaped_rewards_dict = {
            f"agent_{i}": shaped_reward
            for i, shaped_reward in enumerate(shaped_rewards)
        }

        dones = {f"agent_{i}": done for i in range(self.num_agents)}
        dones["__all__"] = done

        return (
            lax.stop_gradient(obs),
            lax.stop_gradient(state),
            rewards,
            dones,
            {"shaped_reward": shaped_rewards_dict},
        )

    # -------------------------------------------------------------------------
    # Agent Movement, Collisions, and Button Interactions
    # -------------------------------------------------------------------------

    def step_agents(
        self,
        key: chex.PRNGKey,
        state: State,
        actions: chex.Array,
    ) -> Tuple[State, float, chex.Array]:
        """Process agent actions and update state."""
        grid = state.grid

        # Movement phase
        def _move_wrapper(agent, action):
            direction = ACTION_TO_DIRECTION[action]

            def _move(agent, dir):
                pos = agent.pos
                new_pos = pos.move_in_bounds(dir, self.width, self.height)

                # Check if new position is walkable
                new_cell_static = grid[new_pos.y, new_pos.x, 0]
                is_walkable = self._is_agent_walkable(
                    new_cell_static, new_pos, state
                )

                new_pos = tree_select(is_walkable, new_pos, pos)
                return agent.replace(pos=new_pos, dir=direction)

            return jax.lax.cond(
                direction != -1,
                _move,
                lambda a, _: a,
                agent,
                direction,
            )

        new_agents = jax.vmap(_move_wrapper)(state.agents, actions)

        # Resolve collisions
        def _masked_positions(mask):
            return tree_select(mask, state.agents.pos, new_agents.pos)

        def _get_collisions(mask):
            positions = _masked_positions(mask)

            collision_grid = jnp.zeros((self.height, self.width))
            collision_grid, _ = jax.lax.scan(
                lambda grid, pos: (grid.at[pos.y, pos.x].add(1), None),
                collision_grid,
                positions,
            )

            collision_mask = collision_grid > 1
            collisions = jax.vmap(lambda p: collision_mask[p.y, p.x])(positions)
            return collisions

        initial_mask = jnp.zeros((self.num_agents,), dtype=bool)
        mask = jax.lax.while_loop(
            lambda mask: jnp.any(_get_collisions(mask)),
            lambda mask: mask | _get_collisions(mask),
            initial_mask,
        )
        new_agents = new_agents.replace(pos=_masked_positions(mask))

        # Prevent swapping
        def _compute_swapped_agents(original_positions, new_positions):
            original_positions = original_positions.to_array()
            new_positions = new_positions.to_array()

            original_pos_expanded = jnp.expand_dims(original_positions, axis=0)
            new_pos_expanded = jnp.expand_dims(new_positions, axis=1)

            swap_mask = (original_pos_expanded == new_pos_expanded).all(axis=-1)
            swap_mask = jnp.fill_diagonal(swap_mask, False, inplace=False)

            swap_pairs = jnp.logical_and(swap_mask, swap_mask.T)
            swapped_agents = jnp.any(swap_pairs, axis=0)
            return swapped_agents

        swap_mask = _compute_swapped_agents(state.agents.pos, new_agents.pos)
        new_agents = new_agents.replace(pos=_masked_positions(swap_mask))

        # Interaction phase
        def _interact_wrapper(carry, x):
            agent, action = x
            is_interact = action == Actions.interact

            def _interact(carry, agent):
                grid, correct_delivery, reward, pot_timers = carry

                (
                    new_grid,
                    new_agent,
                    new_correct_delivery,
                    interact_reward,
                    shaped_reward,
                    new_pot_timers,
                ) = self.process_interact(
                    grid,
                    agent,
                    new_agents.inventory,
                    state.recipe,
                    pot_timers,
                    state.pot_positions,
                    state.pot_active_mask,
                )

                carry = (
                    new_grid,
                    correct_delivery | new_correct_delivery,
                    reward + interact_reward,
                    new_pot_timers,
                )
                return carry, (new_agent, shaped_reward)

            return jax.lax.cond(
                is_interact, _interact, lambda c, a: (c, (a, 0.0)), carry, agent
            )

        carry = (grid, False, 0.0, state.pot_cooking_timer)
        xs = (new_agents, actions)
        (
            (new_grid, new_correct_delivery, reward, new_pot_timers),
            (new_agents, shaped_rewards),
        ) = jax.lax.scan(_interact_wrapper, carry, xs)

        # Update pot timers (cooking and burning)
        new_grid, new_pot_timers = self._update_pot_timers(
            new_grid, new_pot_timers, state.pot_positions, state.pot_active_mask
        )

        # Process button presses
        new_mw_directions = state.moving_wall_directions
        new_mw_paused = state.moving_wall_paused
        new_mw_bounce = state.moving_wall_bounce
        new_btn_toggled = state.button_toggled
        new_barrier_active = state.barrier_active
        new_barrier_timer = state.barrier_timer

        if self.enable_buttons:

            def _process_agent_button(carry, x):
                mw_dirs, mw_paused, mw_bounce, btn_toggled, bar_active, bar_timer = (
                    carry
                )
                agent, action = x
                is_interact = action == Actions.interact
                fwd_pos = agent.get_fwd_pos()
                fwd_static = new_grid[fwd_pos.y, fwd_pos.x, 0]
                is_button = fwd_static == StaticObject.BUTTON

                def _scan_buttons(carry):
                    (
                        mw_dirs,
                        mw_paused,
                        mw_bounce,
                        btn_toggled,
                        bar_active,
                        bar_timer,
                    ) = carry

                    def _check_button(carry, button_idx):
                        (
                            mw_dirs,
                            mw_paused,
                            mw_bounce,
                            btn_toggled,
                            bar_active,
                            bar_timer,
                        ) = carry
                        btn_y = state.button_positions[button_idx, 0]
                        btn_x = state.button_positions[button_idx, 1]
                        is_active = state.button_active_mask[button_idx]
                        is_this = (
                            (btn_y == fwd_pos.y) & (btn_x == fwd_pos.x) & is_active
                        )

                        action_type = state.button_action_type[button_idx]

                        # Toggle button state
                        new_toggled = jax.lax.select(
                            is_this, ~btn_toggled[button_idx], btn_toggled[button_idx]
                        )
                        btn_toggled = btn_toggled.at[button_idx].set(new_toggled)

                        def _apply_target(carry, target_slot):
                            (
                                mw_dirs,
                                mw_paused,
                                mw_bounce,
                                bar_active,
                                bar_timer,
                            ) = carry
                            target_idx = state.button_target_idxs[
                                button_idx, target_slot
                            ]
                            target_enabled = state.button_target_mask[
                                button_idx, target_slot
                            ]
                            should_apply = is_this & target_enabled
                            mw_idx = jnp.clip(target_idx, 0, MAX_MOVING_WALLS - 1)
                            barrier_idx = jnp.clip(target_idx, 0, MAX_BARRIERS - 1)

                            # Moving wall actions
                            mw_paused = jax.lax.select(
                                should_apply
                                & (action_type == ButtonAction.TOGGLE_PAUSE),
                                mw_paused.at[mw_idx].set(~mw_paused[mw_idx]),
                                mw_paused,
                            )

                            new_dir = Direction.opposite(mw_dirs[mw_idx])
                            mw_dirs = jax.lax.select(
                                should_apply
                                & (action_type == ButtonAction.TOGGLE_DIRECTION),
                                mw_dirs.at[mw_idx].set(new_dir),
                                mw_dirs,
                            )

                            mw_bounce = jax.lax.select(
                                should_apply
                                & (action_type == ButtonAction.TOGGLE_BOUNCE),
                                mw_bounce.at[mw_idx].set(~mw_bounce[mw_idx]),
                                mw_bounce,
                            )

                            mw_paused = jax.lax.select(
                                should_apply
                                & (action_type == ButtonAction.TRIGGER_MOVE),
                                mw_paused.at[mw_idx].set(False),
                                mw_paused,
                            )

                            # Barrier actions
                            bar_active = jax.lax.select(
                                should_apply
                                & (action_type == ButtonAction.TOGGLE_BARRIER),
                                bar_active.at[barrier_idx].set(
                                    ~bar_active[barrier_idx]
                                ),
                                bar_active,
                            )

                            bar_active = jax.lax.select(
                                should_apply
                                & (action_type == ButtonAction.TIMED_BARRIER),
                                bar_active.at[barrier_idx].set(False),
                                bar_active,
                            )
                            bar_timer = jax.lax.select(
                                should_apply
                                & (action_type == ButtonAction.TIMED_BARRIER),
                                bar_timer.at[barrier_idx].set(
                                    state.barrier_duration[barrier_idx]
                                ),
                                bar_timer,
                            )

                            return (
                                mw_dirs,
                                mw_paused,
                                mw_bounce,
                                bar_active,
                                bar_timer,
                            ), None

                        (
                            mw_dirs,
                            mw_paused,
                            mw_bounce,
                            bar_active,
                            bar_timer,
                        ), _ = jax.lax.scan(
                            _apply_target,
                            (
                                mw_dirs,
                                mw_paused,
                                mw_bounce,
                                bar_active,
                                bar_timer,
                            ),
                            jnp.arange(MAX_BUTTON_TARGETS),
                        )

                        return (
                            mw_dirs,
                            mw_paused,
                            mw_bounce,
                            btn_toggled,
                            bar_active,
                            bar_timer,
                        ), None

                    (
                        (
                            mw_dirs,
                            mw_paused,
                            mw_bounce,
                            btn_toggled,
                            bar_active,
                            bar_timer,
                        ),
                        _,
                    ) = jax.lax.scan(
                        _check_button,
                        (
                            mw_dirs,
                            mw_paused,
                            mw_bounce,
                            btn_toggled,
                            bar_active,
                            bar_timer,
                        ),
                        jnp.arange(MAX_BUTTONS),
                    )
                    return (
                        mw_dirs,
                        mw_paused,
                        mw_bounce,
                        btn_toggled,
                        bar_active,
                        bar_timer,
                    )

                should_process = is_interact & is_button
                new_carry = jax.lax.cond(
                    should_process,
                    _scan_buttons,
                    lambda c: c,
                    (mw_dirs, mw_paused, mw_bounce, btn_toggled, bar_active, bar_timer),
                )

                return new_carry, None

            (
                (
                    new_mw_directions,
                    new_mw_paused,
                    new_mw_bounce,
                    new_btn_toggled,
                    new_barrier_active,
                    new_barrier_timer,
                ),
                _,
            ) = jax.lax.scan(
                _process_agent_button,
                (
                    new_mw_directions,
                    new_mw_paused,
                    new_mw_bounce,
                    new_btn_toggled,
                    new_barrier_active,
                    new_barrier_timer,
                ),
                (new_agents, actions),
            )

        return (
            state.replace(
                agents=new_agents,
                grid=new_grid,
                pot_cooking_timer=new_pot_timers,
                new_correct_delivery=new_correct_delivery,
                moving_wall_directions=new_mw_directions,
                moving_wall_paused=new_mw_paused,
                moving_wall_bounce=new_mw_bounce,
                button_toggled=new_btn_toggled,
                barrier_active=new_barrier_active,
                barrier_timer=new_barrier_timer,
            ),
            reward,
            shaped_rewards,
        )

    # -------------------------------------------------------------------------
    # Interact Action Handling
    # -------------------------------------------------------------------------

    def process_interact(
        self,
        grid: chex.Array,
        agent: Agent,
        all_inventories: jnp.ndarray,
        recipe: int,
        pot_timers: chex.Array,
        pot_positions: chex.Array,
        pot_active_mask: chex.Array,
    ):
        """Process an interact action for an agent."""
        inventory = agent.inventory
        fwd_pos, fwd_pos_in_bounds = agent.pos.checked_move(
            agent.dir, self.width, self.height
        )

        shaped_reward = jnp.array(0.0, dtype=float)

        interact_cell = grid[fwd_pos.y, fwd_pos.x]
        interact_item = interact_cell[0]
        interact_ingredients = interact_cell[1]
        interact_extra = interact_cell[2]

        plated_recipe = recipe | DynamicObject.PLATE | DynamicObject.COOKED

        # What is the object?
        object_is_plate_pile = fwd_pos_in_bounds & (
            interact_item == StaticObject.PLATE_PILE
        )
        object_is_ingredient_pile = (
            fwd_pos_in_bounds & StaticObject.is_ingredient_pile(interact_item)
        )
        object_is_pile = object_is_plate_pile | object_is_ingredient_pile

        object_is_pot = fwd_pos_in_bounds & (interact_item == StaticObject.POT)
        object_is_goal = fwd_pos_in_bounds & (interact_item == StaticObject.GOAL)
        object_is_wall = fwd_pos_in_bounds & (
            (interact_item == StaticObject.WALL)
            | (interact_item == StaticObject.MOVING_WALL)
        )
        object_is_conveyor = fwd_pos_in_bounds & (
            (interact_item == StaticObject.ITEM_CONVEYOR)
            | (interact_item == StaticObject.PLAYER_CONVEYOR)
        )
        object_has_no_ingredients = interact_ingredients == 0

        # What is in inventory?
        inventory_is_empty = inventory == 0
        inventory_is_ingredient = DynamicObject.is_ingredient(inventory)
        inventory_is_plate = inventory == DynamicObject.PLATE
        inventory_is_dish = (inventory & DynamicObject.COOKED) != 0

        merged_ingredients = interact_ingredients + inventory

        # Pot state
        pot_is_cooked = object_is_pot * (
            (interact_ingredients & DynamicObject.COOKED) != 0
        )
        pot_is_cooking = object_is_pot * (interact_extra > 0) * ~pot_is_cooked
        pot_is_burned = object_is_pot * (
            (interact_ingredients & DynamicObject.BURNED) != 0
        )
        pot_is_idle = object_is_pot * ~pot_is_cooking * ~pot_is_cooked * ~pot_is_burned

        # Check if pot is ready (in burning window)
        # In V3: dish_ready when cooking_timer is between 1 and burn_time
        pot_is_ready = pot_is_cooked

        # Pickup success conditions
        successful_dish_pickup = pot_is_ready * inventory_is_plate
        is_dish_pickup_useful = merged_ingredients == plated_recipe
        if self.shaped_rewards_enabled:
            shaped_reward += (
                successful_dish_pickup
                * is_dish_pickup_useful
                * SHAPED_REWARDS["SOUP_IN_DISH"]
            )

        successful_pickup = (
            object_is_pile * inventory_is_empty
            + successful_dish_pickup
            + object_is_wall * ~object_has_no_ingredients * inventory_is_empty
            + object_is_conveyor * ~object_has_no_ingredients * inventory_is_empty
        )

        # Pot placement
        pot_full = DynamicObject.ingredient_count(interact_ingredients) == 3

        # Check same ingredient type for pot
        pot_ingredient_type = DynamicObject.get_ingredient_type(interact_ingredients)
        inventory_ingredient_type = DynamicObject.get_ingredient_type(inventory)
        same_ingredient_type = (pot_ingredient_type == inventory_ingredient_type) | (
            interact_ingredients == 0
        )

        successful_pot_placement = (
            pot_is_idle * inventory_is_ingredient * ~pot_full * same_ingredient_type
        )
        ingredient_selector = inventory | (inventory << 1)
        is_pot_placement_useful = (interact_ingredients & ingredient_selector) < (
            recipe & ingredient_selector
        )
        if self.shaped_rewards_enabled:
            shaped_reward += (
                successful_pot_placement
                * is_pot_placement_useful
                * SHAPED_REWARDS["PLACEMENT_IN_POT"]
            )

        # Drop on counter/conveyor
        successful_drop = (
            object_is_wall | object_is_conveyor
        ) * object_has_no_ingredients * ~inventory_is_empty + successful_pot_placement

        # Delivery
        successful_delivery = object_is_goal * inventory_is_dish
        no_effect = ~successful_pickup * ~successful_drop * ~successful_delivery

        # Compute new ingredient layer
        pile_ingredient = (
            object_is_plate_pile * DynamicObject.PLATE
            + object_is_ingredient_pile * StaticObject.get_ingredient(interact_item)
        )

        new_ingredients = (
            successful_drop * merged_ingredients + no_effect * interact_ingredients
        )

        # Start cooking when pot becomes full
        pot_full_after_drop = DynamicObject.ingredient_count(new_ingredients) == 3
        auto_cook = pot_is_idle & pot_full_after_drop

        # Update pot timer
        # Find which pot this is
        def _update_pot_timer(pot_idx):
            pot_y, pot_x = pot_positions[pot_idx]
            is_this_pot = (
                (pot_y == fwd_pos.y) & (pot_x == fwd_pos.x) & pot_active_mask[pot_idx]
            )
            new_timer = jax.lax.select(
                is_this_pot & auto_cook, self.pot_cook_time, pot_timers[pot_idx]
            )
            # Reset timer on successful dish pickup
            new_timer = jax.lax.select(
                is_this_pot & successful_dish_pickup, 0, new_timer
            )
            return new_timer

        new_pot_timers = jax.vmap(_update_pot_timer)(jnp.arange(MAX_POTS))

        new_extra = interact_extra  # Keep conveyor directions etc

        new_cell = jnp.array([interact_item, new_ingredients, new_extra])
        new_grid = grid.at[fwd_pos.y, fwd_pos.x].set(new_cell)

        new_inventory = (
            successful_pickup * (pile_ingredient + merged_ingredients)
            + no_effect * inventory
        )
        new_agent = agent.replace(inventory=new_inventory)

        # Reward calculation
        is_correct_recipe = inventory == plated_recipe

        reward = jnp.array(0.0, dtype=float)
        reward += (
            successful_delivery
            * jax.lax.select(is_correct_recipe, 1.0, 0.0)
            * self.delivery_reward
        )

        # Plate pickup reward
        if self.shaped_rewards_enabled:
            inventory_is_plate_now = new_inventory == DynamicObject.PLATE
            successful_plate_pickup = successful_pickup * inventory_is_plate_now
            num_plates_in_inventory = jnp.sum(all_inventories == DynamicObject.PLATE)
            pot_ingredient_counts = jax.vmap(jax.vmap(DynamicObject.ingredient_count))(
                grid[:, :, 1]
            )
            full_unburned_pots = (
                (grid[:, :, 0] == StaticObject.POT)
                & (pot_ingredient_counts == 3)
                & ((grid[:, :, 1] & DynamicObject.BURNED) == 0)
            )
            num_useful_pots = jnp.sum(full_unburned_pots)
            is_plate_pickup_useful = num_plates_in_inventory < num_useful_pots
            shaped_reward += (
                is_plate_pickup_useful
                * successful_plate_pickup
                * SHAPED_REWARDS["PLATE_PICKUP"]
            )

        correct_delivery = successful_delivery & is_correct_recipe

        return (
            new_grid,
            new_agent,
            correct_delivery,
            reward,
            shaped_reward,
            new_pot_timers,
        )

    # -------------------------------------------------------------------------
    # Dynamic Environment Systems
    # -------------------------------------------------------------------------

    def _update_pot_timers(
        self,
        grid: chex.Array,
        pot_timers: chex.Array,
        pot_positions: chex.Array,
        pot_active_mask: chex.Array,
    ) -> Tuple[chex.Array, chex.Array]:
        """Update pot cooking timers and handle burning."""

        def _update_single_pot(carry, pot_idx):
            grid, timers = carry
            pot_y, pot_x = pot_positions[pot_idx]
            is_active = pot_active_mask[pot_idx]
            current_timer = timers[pot_idx]

            pot_cell = grid[pot_y, pot_x]
            pot_ingredients = pot_cell[1]

            # Check if pot is full (has 3 ingredients)
            ingredient_count = DynamicObject.ingredient_count(pot_ingredients)
            pot_is_full = ingredient_count == 3
            pot_is_cooking = (current_timer > 0) & pot_is_full
            pot_already_cooked = (pot_ingredients & DynamicObject.COOKED) != 0

            # Decrement timer if cooking
            new_timer = jax.lax.select(
                is_active & pot_is_cooking, current_timer - 1, current_timer
            )

            # Check if just finished cooking (entered burning window)
            just_finished_cooking = pot_is_cooking & (new_timer == self.pot_burn_time)
            # Mark as cooked when timer reaches burn_time
            new_ingredients = jax.lax.select(
                is_active & just_finished_cooking,
                pot_ingredients | DynamicObject.COOKED,
                pot_ingredients,
            )

            # Check if pot burned (timer hit 0 while cooking)
            just_burned = pot_is_cooking & (new_timer == 0)
            # Reset pot if burned
            new_ingredients = jax.lax.select(
                is_active & just_burned,
                jnp.int32(0),  # Clear pot
                new_ingredients,
            )
            new_timer = jax.lax.select(is_active & just_burned, jnp.int32(0), new_timer)

            # Update grid
            new_cell = jnp.array([pot_cell[0], new_ingredients, pot_cell[2]])
            new_grid = grid.at[pot_y, pot_x].set(new_cell)

            # Update timers
            new_timers = timers.at[pot_idx].set(new_timer)

            return (new_grid, new_timers), None

        (new_grid, new_timers), _ = jax.lax.scan(
            _update_single_pot, (grid, pot_timers), jnp.arange(MAX_POTS)
        )

        return new_grid, new_timers

    def _process_item_conveyors(self, state: State) -> State:
        """Move items on item conveyor belts."""
        if not self.enable_item_conveyors:
            return state

        grid = state.grid

        def _move_item_on_conveyor(grid, conveyor_idx):
            pos = state.item_conveyor_positions[conveyor_idx]
            direction = state.item_conveyor_directions[conveyor_idx]
            is_active = state.item_conveyor_active_mask[conveyor_idx]

            y, x = pos[0], pos[1]
            current_item = grid[y, x, 1]
            has_item = current_item != 0

            # Calculate destination
            dir_vec = DIR_TO_VEC[direction]

            raw_dest_x = x + dir_vec[0]
            raw_dest_y = y + dir_vec[1]
            dest_in_bounds = (
                (raw_dest_x >= 0)
                & (raw_dest_x < self.width)
                & (raw_dest_y >= 0)
                & (raw_dest_y < self.height)
            )
            dest_x = jnp.clip(raw_dest_x, 0, self.width - 1)
            dest_y = jnp.clip(raw_dest_y, 0, self.height - 1)

            # Check if destination can receive item
            dest_static = grid[dest_y, dest_x, 0]
            dest_item = grid[dest_y, dest_x, 1]
            dest_can_receive = (
                dest_in_bounds
                & (
                    (dest_static == StaticObject.WALL)
                    | (dest_static == StaticObject.ITEM_CONVEYOR)
                    | (dest_static == StaticObject.PLAYER_CONVEYOR)
                    | (dest_static == StaticObject.GOAL)
                    | (dest_static == StaticObject.MOVING_WALL)
                )
                & (dest_item == 0)
            )

            should_move = is_active & has_item & dest_can_receive
            should_disappear = is_active & has_item & ~dest_in_bounds

            # Move item
            new_grid = jax.lax.select(
                should_disappear,
                grid.at[y, x, 1].set(0),
                jax.lax.select(
                    should_move,
                    grid.at[y, x, 1].set(0).at[dest_y, dest_x, 1].set(current_item),
                    grid,
                )
            )

            return new_grid, None

        new_grid, _ = jax.lax.scan(
            _move_item_on_conveyor, grid, jnp.arange(MAX_ITEM_CONVEYORS)
        )

        return state.replace(grid=new_grid)

    def _process_player_conveyors(self, state: State) -> State:
        """Push agents on player conveyor belts."""
        if not self.enable_player_conveyors:
            return state

        agents = state.agents
        grid = state.grid

        def _check_agent_on_conveyor(agent_pos, conveyor_idx):
            pos = state.player_conveyor_positions[conveyor_idx]
            is_active = state.player_conveyor_active_mask[conveyor_idx]
            is_on = (agent_pos.x == pos[1]) & (agent_pos.y == pos[0]) & is_active
            return is_on, state.player_conveyor_directions[conveyor_idx]

        def _push_agent(agent):
            # Check all conveyors
            on_conveyor_checks = jax.vmap(
                lambda idx: _check_agent_on_conveyor(agent.pos, idx)
            )(jnp.arange(MAX_PLAYER_CONVEYORS))

            is_on_any, directions = on_conveyor_checks
            # Take first active conveyor's direction
            conveyor_idx = jnp.argmax(is_on_any)
            is_on = jnp.any(is_on_any)
            push_direction = directions[conveyor_idx]

            # Calculate new position
            new_pos = agent.pos.move_in_bounds(push_direction, self.width, self.height)

            # Check if destination is walkable
            dest_static = grid[new_pos.y, new_pos.x, 0]
            dest_walkable = self._is_agent_walkable(dest_static, new_pos, state)

            should_push = is_on & dest_walkable

            final_pos = tree_select(should_push, new_pos, agent.pos)
            return agent.replace(pos=final_pos)

        new_agents = jax.vmap(_push_agent)(agents)

        return state.replace(agents=new_agents)

    def _process_barrier_timers(self, state: State) -> State:
        """Decrement barrier timers and reactivate barriers when timers reach zero."""

        def _update_single_barrier(i):
            is_active_slot = state.barrier_active_mask[i]
            timer = state.barrier_timer[i]
            barrier_active = state.barrier_active[i]

            # Timer only decrements when > 0
            has_timer = timer > 0
            new_timer = jax.lax.select(has_timer, timer - 1, timer)

            # When timer reaches 0, reactivate the barrier
            should_reactivate = is_active_slot & (
                timer == 1
            )  # Will become 0 after decrement
            new_active = jax.lax.select(should_reactivate, True, barrier_active)

            return new_timer, new_active

        # Process all barriers
        new_timers, new_active_states = jax.vmap(_update_single_barrier)(
            jnp.arange(MAX_BARRIERS)
        )

        return state.replace(
            barrier_timer=new_timers,
            barrier_active=new_active_states,
        )

    def _process_moving_walls(self, state: State) -> State:
        """Move moving walls one step in their direction, pushing agents if needed."""
        if not self.enable_moving_walls:
            return state

        grid = state.grid

        # Build agent position arrays for collision checking
        agent_xs = state.agents.pos.x  # [num_agents]
        agent_ys = state.agents.pos.y  # [num_agents]

        def _move_single_wall(carry, wall_idx):
            grid, positions, directions, paused, bounce, ag_xs, ag_ys = carry

            y = positions[wall_idx, 0]
            x = positions[wall_idx, 1]
            direction = directions[wall_idx]
            is_active = state.moving_wall_active_mask[wall_idx]
            is_paused = paused[wall_idx]
            should_process = is_active & ~is_paused

            # Direction vector
            dir_vec = DIR_TO_VEC[direction]
            dest_x = x + dir_vec[0]
            dest_y = y + dir_vec[1]

            # Bounds check
            in_bounds = (
                (dest_x >= 0)
                & (dest_x < self.width)
                & (dest_y >= 0)
                & (dest_y < self.height)
            )

            # Safe indices for array access
            safe_dest_x = jnp.clip(dest_x, 0, self.width - 1)
            safe_dest_y = jnp.clip(dest_y, 0, self.height - 1)

            # Check destination cell
            dest_static = grid[safe_dest_y, safe_dest_x, 0]
            dest_is_empty = dest_static == StaticObject.EMPTY

            # Check if an agent is at the destination
            agent_at_dest = (ag_xs == safe_dest_x) & (ag_ys == safe_dest_y)
            any_agent_at_dest = jnp.any(agent_at_dest)

            # If agent at dest, check if we can push them
            # Beyond position = dest + dir_vec
            beyond_x = safe_dest_x + dir_vec[0]
            beyond_y = safe_dest_y + dir_vec[1]
            beyond_in_bounds = (
                (beyond_x >= 0)
                & (beyond_x < self.width)
                & (beyond_y >= 0)
                & (beyond_y < self.height)
            )
            safe_beyond_x = jnp.clip(beyond_x, 0, self.width - 1)
            safe_beyond_y = jnp.clip(beyond_y, 0, self.height - 1)

            beyond_static = grid[safe_beyond_y, safe_beyond_x, 0]

            # Check if beyond position has an active barrier
            is_barrier_tile = beyond_static == StaticObject.BARRIER
            barrier_blocks = False
            for i in range(MAX_BARRIERS):
                at_barrier_pos = (
                    (state.barrier_positions[i, 0] == safe_beyond_y)
                    & (state.barrier_positions[i, 1] == safe_beyond_x)
                    & state.barrier_active_mask[i]
                )
                barrier_blocks = barrier_blocks | (
                    at_barrier_pos & state.barrier_active[i]
                )

            beyond_walkable = (
                (beyond_static == StaticObject.EMPTY)
                | (beyond_static == StaticObject.ITEM_CONVEYOR)
                | (beyond_static == StaticObject.PLAYER_CONVEYOR)
                | (is_barrier_tile & ~barrier_blocks)
            )
            # Also check no other agent at beyond position
            agent_at_beyond = (ag_xs == safe_beyond_x) & (ag_ys == safe_beyond_y)
            no_agent_at_beyond = ~jnp.any(agent_at_beyond)

            can_push_agent = (
                any_agent_at_dest
                & beyond_in_bounds
                & beyond_walkable
                & no_agent_at_beyond
            )

            # Wall can move if dest is empty (no agent) or if we can push the agent
            # Only allow pushing agents on empty tiles to avoid overwriting static objects
            can_move = (
                should_process
                & in_bounds
                & (
                    (dest_is_empty & ~any_agent_at_dest)
                    | (can_push_agent & (dest_static == StaticObject.EMPTY))
                )
            )

            # Handle bounce: if blocked and bounce enabled, reverse direction
            is_blocked = should_process & ~can_move
            should_bounce = is_blocked & bounce[wall_idx]
            new_direction = jax.lax.select(
                should_bounce,
                Direction.opposite(direction),
                direction,
            )

            # Get item carried by this wall
            old_item = grid[y, x, 1]

            # Clear old position (becomes EMPTY with no item)
            cleared_grid = jax.lax.select(
                can_move,
                grid.at[y, x].set(jnp.array([StaticObject.EMPTY, 0, 0])),
                grid,
            )

            # Set new position with MOVING_WALL + carried item + direction in extra channel
            new_cell = jnp.array([StaticObject.MOVING_WALL, old_item, new_direction])
            moved_grid = jax.lax.select(
                can_move,
                cleared_grid.at[safe_dest_y, safe_dest_x].set(new_cell),
                cleared_grid,
            )

            # Update direction in grid at current position if staying (bounce case)
            final_grid = jax.lax.select(
                ~can_move & should_bounce,
                moved_grid.at[y, x, 2].set(new_direction),
                moved_grid,
            )

            # Update position array
            new_y = jax.lax.select(can_move, safe_dest_y, y)
            new_x = jax.lax.select(can_move, safe_dest_x, x)
            new_positions = positions.at[wall_idx].set(jnp.array([new_y, new_x]))

            # Update direction array
            new_directions = directions.at[wall_idx].set(new_direction)

            # Push agent: update agent positions if we pushed
            # Find which agent was pushed (first match)
            push_agent_idx = jnp.argmax(agent_at_dest)
            new_ag_xs = jax.lax.select(
                can_move & can_push_agent,
                ag_xs.at[push_agent_idx].set(safe_beyond_x),
                ag_xs,
            )
            new_ag_ys = jax.lax.select(
                can_move & can_push_agent,
                ag_ys.at[push_agent_idx].set(safe_beyond_y),
                ag_ys,
            )

            return (
                final_grid,
                new_positions,
                new_directions,
                paused,
                bounce,
                new_ag_xs,
                new_ag_ys,
            ), None

        init_carry = (
            grid,
            state.moving_wall_positions,
            state.moving_wall_directions,
            state.moving_wall_paused,
            state.moving_wall_bounce,
            agent_xs,
            agent_ys,
        )

        (
            (
                new_grid,
                new_positions,
                new_directions,
                new_paused,
                _bounce,
                new_ag_xs,
                new_ag_ys,
            ),
            _,
        ) = jax.lax.scan(_move_single_wall, init_carry, jnp.arange(MAX_MOVING_WALLS))

        # Re-pause walls linked to TRIGGER_MOVE buttons
        def _reapply_trigger_pause(paused, button_idx):
            is_active = state.button_active_mask[button_idx]
            is_trigger = (
                state.button_action_type[button_idx] == ButtonAction.TRIGGER_MOVE
            )

            def _pause_target(paused, target_slot):
                target_idx = state.button_target_idxs[button_idx, target_slot]
                target_enabled = state.button_target_mask[button_idx, target_slot]
                mw_idx = jnp.clip(target_idx, 0, MAX_MOVING_WALLS - 1)
                new_paused = jax.lax.select(
                    is_active & is_trigger & target_enabled,
                    paused.at[mw_idx].set(True),
                    paused,
                )
                return new_paused, None

            paused, _ = jax.lax.scan(
                _pause_target, paused, jnp.arange(MAX_BUTTON_TARGETS)
            )
            return paused, None

        new_paused, _ = jax.lax.scan(
            _reapply_trigger_pause, new_paused, jnp.arange(MAX_BUTTONS)
        )

        # Rebuild agents with updated positions
        new_agents = state.agents.replace(pos=Position(x=new_ag_xs, y=new_ag_ys))

        return state.replace(
            grid=new_grid,
            agents=new_agents,
            moving_wall_positions=new_positions,
            moving_wall_directions=new_directions,
            moving_wall_paused=new_paused,
        )

    # -------------------------------------------------------------------------
    # Order Queue
    # -------------------------------------------------------------------------

    def _process_order_queue(
        self, state: State, key: chex.PRNGKey
    ) -> Tuple[State, float]:
        """Process order queue: generate new orders, check expirations."""
        if not self.enable_order_queue:
            return state, 0.0

        order_types = state.order_types
        order_expirations = state.order_expirations
        order_active_mask = state.order_active_mask

        # Decrement expirations
        new_expirations = jnp.where(
            order_active_mask, order_expirations - 1, order_expirations
        )

        # Check for expired orders
        expired_mask = order_active_mask & (new_expirations <= 0)
        num_expired = jnp.sum(expired_mask)
        reward = num_expired * ORDER_EXPIRED_PENALTY

        # Deactivate expired orders
        new_active_mask = order_active_mask & ~expired_mask

        # Maybe generate new order
        key, subkey = jax.random.split(key)
        should_generate = jax.random.uniform(subkey) < self.order_generation_rate

        # Find first empty slot
        empty_slots = ~new_active_mask
        first_empty_idx = jnp.argmax(empty_slots)
        has_empty_slot = jnp.any(empty_slots)

        # Generate random order type (1 = onion soup, 2 = tomato soup if num_ingredients > 1)
        key, subkey = jax.random.split(key)
        new_order_type = jax.random.randint(
            subkey, (), 1, min(self.layout.num_ingredients + 1, 3)
        )

        should_add = should_generate & has_empty_slot
        new_order_types = jax.lax.select(
            should_add, order_types.at[first_empty_idx].set(new_order_type), order_types
        )
        new_expirations = jax.lax.select(
            should_add,
            new_expirations.at[first_empty_idx].set(self.order_expiration_time),
            new_expirations,
        )
        new_active_mask = jax.lax.select(
            should_add, new_active_mask.at[first_empty_idx].set(True), new_active_mask
        )

        return state.replace(
            order_types=new_order_types,
            order_expirations=new_expirations,
            order_active_mask=new_active_mask,
        ), reward

    # -------------------------------------------------------------------------
    # Termination, Observations, and Spaces
    # -------------------------------------------------------------------------

    def is_terminal(self, state: State) -> bool:
        """Check whether state is terminal."""
        done_steps = state.time >= self.max_steps
        return done_steps | state.terminal

    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        """Get observations for all agents."""
        if not isinstance(self.observation_type, list):
            return self.get_obs_for_type(state, self.observation_type)

        all_obs = {}
        for i, obs_type in enumerate(self.observation_type):
            obs = self.get_obs_for_type(state, obs_type)
            key = f"agent_{i}"
            all_obs[key] = obs[key]
        return all_obs

    def get_obs_for_type(
        self, state: State, obs_type: ObservationType
    ) -> Dict[str, chex.Array]:
        """Get observations for a specific observation type."""
        if obs_type == ObservationType.DEFAULT:
            all_obs = self.get_obs_default(state)
        elif obs_type == ObservationType.FEATURIZED:
            # Simplified placeholder
            all_obs = jnp.zeros((self.num_agents,) + self.obs_shape)
        else:
            raise ValueError(f"Invalid observation type: {obs_type}")

        def _mask_obs(obs, agent):
            view_size = self.agent_view_size
            pos = agent.pos

            padded_obs = jnp.pad(
                obs,
                ((view_size, view_size), (view_size, view_size), (0, 0)),
                mode="constant",
                constant_values=0,
            )

            sliced_obs = jax.lax.dynamic_slice(
                padded_obs,
                (pos.y, pos.x, 0),
                self.obs_shape,
            )

            return sliced_obs

        if self.agent_view_size is not None:
            all_obs = jax.vmap(_mask_obs)(all_obs, state.agents)

        return {f"agent_{i}": obs for i, obs in enumerate(all_obs)}

    def get_obs_default(self, state: State) -> chex.Array:
        """Get default grid-based observations."""
        width = self.width
        height = self.height
        num_ingredients = self.layout.num_ingredients

        static_objects = state.grid[:, :, 0]
        ingredients = state.grid[:, :, 1]
        static_encoding = jnp.array([
            StaticObject.WALL,
            StaticObject.GOAL,
            StaticObject.POT,
            StaticObject.RECIPE_INDICATOR,
            StaticObject.PLATE_PILE,
            StaticObject.ITEM_CONVEYOR,
            StaticObject.PLAYER_CONVEYOR,
            StaticObject.MOVING_WALL,
            StaticObject.BUTTON,
            StaticObject.BARRIER,
        ])
        static_layers = static_objects[..., None] == static_encoding

        def _ingredient_layers(ingredients):
            shift = jnp.array([0, 1] + [2 * (i + 1) for i in range(num_ingredients)])
            mask = jnp.array([0x1, 0x1] + [0x3] * num_ingredients)

            layers = ingredients[..., None] >> shift
            layers = layers & mask
            return layers

        recipe_indicator_mask = static_objects == StaticObject.RECIPE_INDICATOR
        recipe_ingredients = jnp.where(recipe_indicator_mask, state.recipe, 0)

        pot_timer_layer = jnp.zeros((height, width), dtype=jnp.int32)
        for i in range(MAX_POTS):
            y, x = state.pot_positions[i]
            timer = state.pot_cooking_timer[i]
            is_active = state.pot_active_mask[i]
            pot_timer_layer = jax.lax.select(
                is_active, pot_timer_layer.at[y, x].set(timer), pot_timer_layer
            )

        extra_layers = jnp.stack([pot_timer_layer], axis=-1)

        def _agent_layers(agent):
            pos = agent.pos
            direction = agent.dir
            inv = agent.inventory

            pos_layers = (
                jnp.zeros((height, width, 1), dtype=jnp.uint8)
                .at[pos.y, pos.x, 0]
                .set(1)
            )
            dir_layers = (
                jnp.zeros((height, width, 4), dtype=jnp.uint8)
                .at[pos.y, pos.x, direction]
                .set(1)
            )
            inv_grid = jnp.zeros_like(ingredients).at[pos.y, pos.x].set(inv)
            inv_layers = _ingredient_layers(inv_grid)

            return jnp.concatenate([pos_layers, dir_layers, inv_layers], axis=-1)

        def _agent_obs(agent_id):
            agent_layers = jax.vmap(_agent_layers)(state.agents)
            agent_layer = agent_layers[agent_id]
            all_agent_layers = jnp.sum(agent_layers, axis=0)
            other_agent_layers = all_agent_layers - agent_layer

            ingredients_layers = _ingredient_layers(ingredients)
            recipe_layers = _ingredient_layers(recipe_ingredients)

            ingredient_pile_encoding = jnp.array(
                [StaticObject.INGREDIENT_PILE_BASE + i for i in range(num_ingredients)]
            )
            ingredient_pile_layers = (
                static_objects[..., None] == ingredient_pile_encoding
            )

            return jnp.concatenate(
                [
                    agent_layer,
                    other_agent_layers,
                    static_layers,
                    ingredient_pile_layers,
                    ingredients_layers,
                    recipe_layers,
                    extra_layers,
                ],
                axis=-1,
            )

        return jax.vmap(_agent_obs)(jnp.arange(self.num_agents))

    @property
    def name(self) -> str:
        """Environment name."""
        return "Overcooked V3"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return len(self.action_set)

    def action_space(self, agent_id="") -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(len(self.action_set), dtype=jnp.uint32)

    def observation_space(self, agent_id="") -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(0, 255, self.obs_shape)
