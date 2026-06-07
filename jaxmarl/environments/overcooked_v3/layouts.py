"""Layout definitions and parsing for Overcooked V3.

DESIGN NOTES:
- don't make item conveyor belts / player conveyor belts move things to the same destination - this will cause race conditions and maybe make the items disappear.
"""

from jaxmarl.environments.overcooked_v3.common import (
    StaticObject,
    Direction,
    ButtonAction,
)
from jaxmarl.environments.overcooked_v3.settings import (
    MAX_POTS,
    MAX_ITEM_CONVEYORS,
    MAX_PLAYER_CONVEYORS,
    MAX_MOVING_WALLS,
    MAX_BUTTONS,
    MAX_BARRIERS,
    MAX_BUTTON_TARGETS,
)
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
import itertools


# Standard layouts from Overcooked-AI
cramped_room = """
WWPWW
OA AO
W   W
WBWXW
"""

asymm_advantages = """
WWWWWWWWW
O WXWOW X
W   P   W
W A PA  W
WWWBWBWWW
"""

coord_ring = """
WWWPW
W A P
BAW W
O   W
WOXWW
"""

forced_coord = """
WWWPW
O WAP
OAW W
B W W
WWWXW
"""

counter_circuit = """
WWWPPWWW
W A    W
B WWWW X
W     AW
WWWOOWWW
"""

# Layouts with recipe indicators
cramped_room_v2 = """
WWPWW
0A A1
W   R
WBWXW
"""

# New layout with conveyor belts (example)
conveyor_demo = """
WWPWWWW
0A >  X
W  v  W
W  > AW
WWBWWWW
"""

# Player conveyor demo
player_conveyor_demo = """
WWWWWWW
0A ]  X
W     W
W   [ P
WWWBWWW
"""

# Player conveyor loop - 2x2 clockwise loop for testing
# ] pushes right, } pushes down, [ pushes left, { pushes up
player_conveyor_loop = """
WWPBW
W]}AW
W{[0W
WWXWW
"""

race_against_the_clock = """
XWWWWWWWWWW 2
            1
 WWWWWWWWWW 0
AB        WWW
PW          W
AB        WWW
 WWWWWWWWWW 0
            1
XWWWWWWWWWW 2
"""

maze_conveyor_hell  = """
01 W   W WWW
A  W W W  WW
vW W W W  WW
vW   W    Wv
vWWWWWWWW Wv
vW>>>>>>> Wv
vW WWWWWWPWv
A       WXW 
B          W
"""

coordinated_temporal_conveyor = """
>>>>>>vW   X
      vW  A 
     WvW    
     Wv   PB
  A  WvWWWWW
01    v>>>>W
"""

general_conveyor_level_1 = """
012    P
 A     W
       W
]]]]]]]]
[[[[[[[[
       W
      AW
BX     P
"""

general_conveyor_level_2 = """
W W   1WWWW
0  WW WW  P
2A  ]]]   P
 A  W W    
   WW WW   
    [[[   B
   WW WW  B
   XW W    
   XW WWWWW
"""

general_conveyor_level_3 = """
A 01WWW
A]]]]}2
W{WWW}W
W{[[[[W
WBWWPWW
"""

middle_conveyor = """
WWWWW^WWWWW
WW  W^W  WW
WW AW^WA WW
W1   ^   PW
WW  W^W  WW
WW BW^WB WW
W0   ^   XW
WW  W^W  WW
WW  W^W  WW
WWWWW^WWWWW
"""


follow_the_leader = """
WWWWWWWW
WWB1  WW
W0AWA PW
W  W  WW
WWWW  XW
W     WW
WWWWWWWW
"""

around_the_island = """
WW0W1WWWWW
B        W
W  A     W
WWWWWWW  X
W  A     W
W        W
WWWPWWWWWW
"""

single_file = """
WBWWPWW
W A A W
W WWW W
X     W
WW1W0WW
"""


# Moving wall demo - wall moves down, button reverses its direction
moving_wall_demo = """
WWWPWWW
0As   X
W  !  W
W    AW
WWWBWWW
"""

# Moving wall bounce demo - two walls bouncing back and forth
moving_wall_bounce_demo = """
WWWWPWWWWW
0A e   AXW
W        W
W  e !   W
WWWWBWWWWW
"""

# Barrier demo - togglable barriers that block all directions
barrier_demo = """
WWWPWWW
0A #  X
W  #  W
W    AW
WWWBWWW
"""

# Timed barrier demo - button deactivates barrier temporarily
timed_barrier_demo = """
WWWPWWW
0A #  X
W ! ! W
W  # AW
WWWBWWW
"""

# Mixed button demo - one button controls a moving wall, the other controls a barrier
moving_wall_barrier_button_demo = """
WWWWWWWW
W0A ! sW
W P #  W
W B XA!W
WWWWWWWW
"""


@dataclass
class Layout:
    """Layout definition for Overcooked V3."""
    # Agent positions: list of (x, y) tuples
    agent_positions: List[Tuple[int, int]]

    # height x width grid with static items
    static_objects: np.ndarray

    # Number of unique ingredient types
    num_ingredients: int

    # Possible recipes (list of lists of ingredient indices)
    possible_recipes: Optional[List[List[int]]]

    # Conveyor belt information
    # Item conveyors: list of (y, x, direction) tuples
    item_conveyor_info: List[Tuple[int, int, int]] = field(default_factory=list)

    # Player conveyors: list of (y, x, direction) tuples
    player_conveyor_info: List[Tuple[int, int, int]] = field(default_factory=list)

    # Moving walls: list of (y, x, direction, bounce) tuples
    moving_wall_info: List[Tuple[int, int, int, bool]] = field(default_factory=list)

    # Buttons: list of (y, x, target_idxs, action_type) tuples
    button_info: List[Tuple[int, int, Tuple[int, ...], int]] = field(
        default_factory=list
    )

    # Barriers: list of (y, x, active) tuples
    barrier_info: List[Tuple[int, int, bool]] = field(default_factory=list)

    def __post_init__(self):
        if len(self.agent_positions) == 0:
            raise ValueError("At least one agent position must be provided")
        if self.num_ingredients < 1:
            raise ValueError("At least one ingredient must be available")
        if self.possible_recipes is None:
            self.possible_recipes = self._get_all_possible_recipes(self.num_ingredients)

    @property
    def height(self):
        return self.static_objects.shape[0]

    @property
    def width(self):
        return self.static_objects.shape[1]

    def to_string(self) -> str:
        """Convert this Layout back to its string representation.

        Returns:
            String representation suitable for Layout.from_string()
        """
        height, width = self.static_objects.shape

        grid = [[' ' for _ in range(width)] for _ in range(height)]

        static_to_symbol = {
            StaticObject.WALL: 'W',
            StaticObject.GOAL: 'X',
            StaticObject.PLATE_PILE: 'B',
            StaticObject.POT: 'P',
            StaticObject.RECIPE_INDICATOR: 'R',
        }

        item_conveyor_symbols = {
            Direction.RIGHT: '>',
            Direction.LEFT: '<',
            Direction.UP: '^',
            Direction.DOWN: 'v',
        }

        player_conveyor_symbols = {
            Direction.RIGHT: ']',
            Direction.LEFT: '[',
            Direction.UP: '{',
            Direction.DOWN: '}',
        }

        moving_wall_symbols = {
            Direction.RIGHT: 'e',
            Direction.LEFT: 'w',
            Direction.UP: 'n',
            Direction.DOWN: 's',
        }

        item_conveyors = {(y, x): direction for y, x, direction in self.item_conveyor_info}
        player_conveyors = {(y, x): direction for y, x, direction in self.player_conveyor_info}
        moving_walls = {
            (y, x): direction for y, x, direction, _ in self.moving_wall_info
        }
        buttons = {(y, x) for y, x, _, _ in self.button_info}
        barriers = {(y, x) for y, x, _ in self.barrier_info}

        for y in range(height):
            for x in range(width):
                obj = self.static_objects[y, x]

                if (y, x) in item_conveyors:
                    direction = Direction(item_conveyors[(y, x)])
                    grid[y][x] = item_conveyor_symbols[direction]
                elif (y, x) in player_conveyors:
                    direction = Direction(player_conveyors[(y, x)])
                    grid[y][x] = player_conveyor_symbols[direction]
                elif (y, x) in moving_walls:
                    direction = Direction(moving_walls[(y, x)])
                    grid[y][x] = moving_wall_symbols[direction]
                elif (y, x) in buttons:
                    grid[y][x] = '!'
                elif (y, x) in barriers:
                    grid[y][x] = '#'
                elif StaticObject.is_ingredient_pile(obj):
                    ingredient_idx = obj - StaticObject.INGREDIENT_PILE_BASE
                    grid[y][x] = str(ingredient_idx)
                elif obj in static_to_symbol:
                    grid[y][x] = static_to_symbol[obj]

        for agent_x, agent_y in self.agent_positions:
            grid[agent_y][agent_x] = 'A'

        lines = [''.join(row) for row in grid]
        return '\n' + '\n'.join(lines) + '\n'

    def get_info(self) -> dict:
        """Get summary information about this layout.

        Returns:
            Dictionary with layout statistics
        """
        info = {
            'dimensions': (self.width, self.height),
            'num_agents': len(self.agent_positions),
            'num_pots': 0,
            'num_ingredient_piles': {},
            'num_plate_piles': 0,
            'num_goals': 0,
            'num_walls': 0,
            'num_item_conveyors': len(self.item_conveyor_info),
            'num_player_conveyors': len(self.player_conveyor_info),
            'has_recipe_indicator': False,
            'possible_recipes': self.possible_recipes,
        }

        for y in range(self.height):
            for x in range(self.width):
                obj = self.static_objects[y, x]

                if obj == StaticObject.POT:
                    info['num_pots'] += 1
                elif obj == StaticObject.PLATE_PILE:
                    info['num_plate_piles'] += 1
                elif obj == StaticObject.GOAL:
                    info['num_goals'] += 1
                elif obj == StaticObject.WALL:
                    info['num_walls'] += 1
                elif obj == StaticObject.RECIPE_INDICATOR:
                    info['has_recipe_indicator'] = True
                elif StaticObject.is_ingredient_pile(obj):
                    ingredient_idx = obj - StaticObject.INGREDIENT_PILE_BASE
                    if ingredient_idx not in info['num_ingredient_piles']:
                        info['num_ingredient_piles'][ingredient_idx] = 0
                    info['num_ingredient_piles'][ingredient_idx] += 1

        return info

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate this layout for common issues.

        Returns:
            Tuple of (is_valid, list of error/warning messages)
        """
        errors = []
        warnings = []

        info = self.get_info()

        if info['num_agents'] == 0:
            errors.append("Layout must have at least one agent")

        if info['num_goals'] == 0:
            errors.append("Layout must have at least one delivery zone (goal)")

        if len(info['num_ingredient_piles']) == 0:
            errors.append("Layout must have at least one ingredient pile")

        if info['num_plate_piles'] == 0:
            warnings.append("No plate pile found - agents won't be able to serve soup")

        if info['num_pots'] == 0:
            warnings.append("No pots found - agents won't be able to cook")

        if info['num_pots'] > MAX_POTS:
            errors.append(f"Too many pots ({info['num_pots']} > {MAX_POTS}). Increase MAX_POTS in settings.py")

        if info['num_item_conveyors'] > MAX_ITEM_CONVEYORS:
            errors.append(f"Too many item conveyors ({info['num_item_conveyors']} > {MAX_ITEM_CONVEYORS})")

        if info['num_player_conveyors'] > MAX_PLAYER_CONVEYORS:
            errors.append(f"Too many player conveyors ({info['num_player_conveyors']} > {MAX_PLAYER_CONVEYORS})")

        if len(self.moving_wall_info) > MAX_MOVING_WALLS:
            errors.append(
                f"Too many moving walls ({len(self.moving_wall_info)} > {MAX_MOVING_WALLS})"
            )

        if len(self.button_info) > MAX_BUTTONS:
            errors.append(
                f"Too many buttons ({len(self.button_info)} > {MAX_BUTTONS})"
            )

        if len(self.barrier_info) > MAX_BARRIERS:
            errors.append(
                f"Too many barriers ({len(self.barrier_info)} > {MAX_BARRIERS})"
            )

        moving_wall_actions = {
            ButtonAction.TOGGLE_PAUSE,
            ButtonAction.TOGGLE_DIRECTION,
            ButtonAction.TOGGLE_BOUNCE,
            ButtonAction.TRIGGER_MOVE,
        }
        barrier_actions = {
            ButtonAction.TOGGLE_BARRIER,
            ButtonAction.TIMED_BARRIER,
        }

        def _in_bounds(y, x):
            return 0 <= y < self.height and 0 <= x < self.width

        for idx, (y, x, direction, _) in enumerate(self.moving_wall_info):
            if not _in_bounds(y, x):
                errors.append(
                    f"Moving wall {idx} position {(y, x)} is outside layout bounds"
                )
                continue
            if self.static_objects[y, x] != StaticObject.MOVING_WALL:
                errors.append(
                    f"Moving wall {idx} at {(y, x)} is not encoded as a moving wall tile"
                )
            try:
                Direction(direction)
            except (TypeError, ValueError):
                errors.append(f"Moving wall {idx} has invalid direction {direction!r}")

        for idx, (y, x, target_idxs, action_type) in enumerate(self.button_info):
            if not _in_bounds(y, x):
                errors.append(f"Button {idx} position {(y, x)} is outside layout bounds")
                continue
            if self.static_objects[y, x] != StaticObject.BUTTON:
                errors.append(
                    f"Button {idx} at {(y, x)} is not encoded as a button tile"
                )

            try:
                action = ButtonAction(action_type)
            except (TypeError, ValueError):
                errors.append(f"Button {idx} has invalid action type {action_type!r}")
                continue

            if isinstance(target_idxs, list):
                target_idxs = tuple(target_idxs)
            elif not isinstance(target_idxs, tuple):
                target_idxs = (target_idxs,)

            if len(target_idxs) == 0:
                errors.append(f"Button {idx} must target at least one index")
                continue

            if len(target_idxs) > MAX_BUTTON_TARGETS:
                errors.append(
                    f"Button {idx} targets {len(target_idxs)} indexes, but at most "
                    f"{MAX_BUTTON_TARGETS} are supported"
                )
                continue

            for target_idx in target_idxs:
                try:
                    target_idx = int(target_idx)
                except (TypeError, ValueError):
                    errors.append(
                        f"Button {idx} target index {target_idx!r} must be an integer"
                    )
                    continue

                if action in moving_wall_actions:
                    if target_idx < 0 or target_idx >= len(self.moving_wall_info):
                        errors.append(
                            f"Button {idx} targets moving wall {target_idx}, but only "
                            f"{len(self.moving_wall_info)} moving walls exist"
                        )
                elif action in barrier_actions:
                    if target_idx < 0 or target_idx >= len(self.barrier_info):
                        errors.append(
                            f"Button {idx} targets barrier {target_idx}, but only "
                            f"{len(self.barrier_info)} barriers exist"
                        )

        for idx, (y, x, _) in enumerate(self.barrier_info):
            if not _in_bounds(y, x):
                errors.append(
                    f"Barrier {idx} position {(y, x)} is outside layout bounds"
                )
                continue
            if self.static_objects[y, x] != StaticObject.BARRIER:
                errors.append(
                    f"Barrier {idx} at {(y, x)} is not encoded as a barrier tile"
                )

        if self.possible_recipes is None:
            if not info['has_recipe_indicator']:
                errors.append("Layout has no recipe indicator and no possible_recipes specified")
        elif not isinstance(self.possible_recipes, list):
            errors.append("possible_recipes must be a list")
        elif len(self.possible_recipes) == 0:
            if not info['has_recipe_indicator']:
                errors.append("Layout has no recipe indicator and no possible_recipes specified")
        else:
            for i, recipe in enumerate(self.possible_recipes):
                if not isinstance(recipe, list) or len(recipe) != 3:
                    errors.append(f"Recipe {i} must be a list of exactly 3 ingredient indices")
                else:
                    for ingredient_idx in recipe:
                        try:
                            ingredient_idx = int(ingredient_idx)
                        except (TypeError, ValueError):
                            errors.append(
                                f"Recipe {i} ingredient {ingredient_idx!r} must be an integer index"
                            )
                            continue
                        if ingredient_idx < 0:
                            errors.append(
                                f"Recipe {i} ingredient {ingredient_idx} must be non-negative"
                            )
                        elif ingredient_idx not in info['num_ingredient_piles'] and ingredient_idx < self.num_ingredients:
                            warnings.append(f"Recipe uses ingredient {ingredient_idx} but no pile exists in layout")

        all_messages = errors + warnings
        is_valid = len(errors) == 0

        return is_valid, all_messages

    @staticmethod
    def _is_agent_walkable_tile(obj) -> bool:
        return obj in (StaticObject.EMPTY, StaticObject.PLAYER_CONVEYOR)

    @staticmethod
    def _is_interaction_access_tile(obj) -> bool:
        return obj in (
            StaticObject.EMPTY,
            StaticObject.PLAYER_CONVEYOR,
            StaticObject.BARRIER,
        )

    def _has_adjacent_walkable_tile(self, y: int, x: int) -> bool:
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            adj_y = y + dy
            adj_x = x + dx
            if not (0 <= adj_y < self.height and 0 <= adj_x < self.width):
                continue
            if self._is_interaction_access_tile(self.static_objects[adj_y, adj_x]):
                return True
        return False

    def _has_interactable_object(self, target_obj) -> bool:
        for y in range(self.height):
            for x in range(self.width):
                if (
                    self.static_objects[y, x] == target_obj
                    and self._has_adjacent_walkable_tile(y, x)
                ):
                    return True
        return False

    def _has_interactable_ingredient(self, ingredient_idx: int) -> bool:
        target_obj = StaticObject.INGREDIENT_PILE_BASE + ingredient_idx
        return self._has_interactable_object(target_obj)

    def validate_playable(
        self,
        enforce_same_ingredient_recipes: bool = True,
    ) -> Tuple[bool, List[str]]:
        """Validate that the layout can support a soup-delivery episode.

        This is stricter than validate(): missing pots/plates and impossible
        recipes are fatal for training, even if they can be useful in demos.
        """
        errors = []
        is_valid, base_messages = self.validate()
        if not is_valid:
            errors.extend(base_messages)

        info = self.get_info()
        ingredient_piles = set(info["num_ingredient_piles"].keys())

        if info["num_agents"] < 1:
            errors.append("Layout needs at least one agent")
        if len(set(self.agent_positions)) != len(self.agent_positions):
            errors.append(
                f"Agent starting positions must be unique: {self.agent_positions}"
            )
        for agent_x, agent_y in self.agent_positions:
            if not (0 <= agent_y < self.height and 0 <= agent_x < self.width):
                errors.append(
                    f"Agent start {(agent_x, agent_y)} is outside layout bounds"
                )
            elif not self._is_agent_walkable_tile(self.static_objects[agent_y, agent_x]):
                errors.append(
                    f"Agent start {(agent_x, agent_y)} is not on a walkable tile"
                )

        if info["num_pots"] < 1:
            errors.append("Layout needs at least one pot")
        elif not self._has_interactable_object(StaticObject.POT):
            errors.append("Layout needs at least one pot adjacent to a walkable tile")

        if info["num_plate_piles"] < 1:
            errors.append("Layout needs at least one plate pile")
        elif not self._has_interactable_object(StaticObject.PLATE_PILE):
            errors.append("Layout needs at least one plate pile adjacent to a walkable tile")

        if info["num_goals"] < 1:
            errors.append("Layout needs at least one delivery zone")
        elif not self._has_interactable_object(StaticObject.GOAL):
            errors.append("Layout needs at least one delivery zone adjacent to a walkable tile")

        if not ingredient_piles:
            errors.append("Layout needs at least one ingredient pile")

        if not self.possible_recipes:
            errors.append("Layout needs at least one possible recipe")
        else:
            for recipe in self.possible_recipes:
                if not isinstance(recipe, list) or len(recipe) != 3:
                    continue

                missing = sorted(set(recipe) - ingredient_piles)
                if missing:
                    errors.append(
                        f"Recipe {recipe} requires missing ingredient piles: {missing}"
                    )

                blocked_ingredients = [
                    ingredient_idx
                    for ingredient_idx in sorted(set(recipe))
                    if ingredient_idx in ingredient_piles
                    and not self._has_interactable_ingredient(ingredient_idx)
                ]
                if blocked_ingredients:
                    errors.append(
                        f"Recipe {recipe} requires boxed-in ingredient piles: {blocked_ingredients}"
                    )

                if enforce_same_ingredient_recipes and len(set(recipe)) != 1:
                    errors.append(
                        f"Recipe {recipe} is mixed, but current pot logic only supports same-ingredient soups"
                    )

        item_conveyor_destinations = {}
        direction_to_delta = {
            Direction.UP: (-1, 0),
            Direction.DOWN: (1, 0),
            Direction.RIGHT: (0, 1),
            Direction.LEFT: (0, -1),
        }

        for y, x, direction in self.item_conveyor_info:
            if self.static_objects[y, x] != StaticObject.ITEM_CONVEYOR:
                errors.append(
                    f"Item conveyor at {(y, x)} is not encoded as an item conveyor tile"
                )
                continue

            dy, dx = direction_to_delta[Direction(direction)]
            dest = (y + dy, x + dx)
            item_conveyor_destinations.setdefault(dest, []).append((y, x))

        for dest, sources in item_conveyor_destinations.items():
            if len(sources) > 1:
                errors.append(
                    f"Multiple item conveyors target {dest}: {sources}"
                )

        for y, x, _ in self.player_conveyor_info:
            if self.static_objects[y, x] != StaticObject.PLAYER_CONVEYOR:
                errors.append(
                    f"Player conveyor at {(y, x)} is not encoded as a player conveyor tile"
                )

        return len(errors) == 0, errors

    @staticmethod
    def annotate_layout_string(layout_string: str) -> str:
        """Add annotations to a layout string explaining the symbols.

        Args:
            layout_string: Layout string to annotate

        Returns:
            Annotated string with legend
        """
        legend = """
            Symbol Legend:
            W = Wall/Counter
            P = Pot
            B = Plate (Bowl) Pile
            X = Delivery Zone (Goal)
            A = Agent Start Position
            R = Recipe Indicator (randomized recipes)
            0-9 = Ingredient Piles (0=onion, 1=tomato, 2=lettuce, etc.)
            
            Item Conveyors (move items):
                > = moves right
                < = moves left
                ^ = moves up
                v = moves down
            
            Player Conveyors (push agents):
                ] = pushes right
                [ = pushes left
                { = pushes up
                } = pushes down
            
            [space] = Empty walkable floor

            Layout:
        """
        return legend + layout_string

    @staticmethod
    def _get_all_possible_recipes(num_ingredients: int) -> List[List[int]]:
        """Get all possible recipes given the number of ingredients."""
        available_ingredients = list(range(num_ingredients)) * 3
        raw_combinations = itertools.combinations(available_ingredients, 3)
        unique_recipes = set(
            tuple(sorted(combination)) for combination in raw_combinations
        )
        return [list(recipe) for recipe in unique_recipes]

    @staticmethod
    def from_string(
        grid,
        possible_recipes=None,
        swap_agents=False,
        moving_wall_bounce=None,
        button_config=None,
        barrier_config=None,
        strict_rectangular=True
    ):
        """Parse a string representation of the layout.

        Symbols:
            W: wall
            A: agent
            X: goal (delivery zone)
            B: plate (bowl) pile
            P: pot location
            R: recipe of the day indicator
            0-9: Ingredient x pile
            ' ' (space): empty cell

            Item conveyor belts (move items):
            >: right
            <: left
            ^: up
            v: down

            Player conveyor belts (push agents):
            ]: right
            [: left
            {: up
            }: down

            Moving walls (move in direction each step):
            n: up
            s: down
            e: right
            w: left (west)

            Buttons (interact to trigger linked wall action):
            !: button (linked to wall by button_config)

            Barriers (togglable blocking tiles):
            #: barrier (blocks all movement when active)

        Args:
            grid: ASCII string layout
            possible_recipes: List of recipes, or None for auto-detect
            swap_agents: Reverse agent order
            moving_wall_bounce: List of bools per moving wall. Parse order is
                row-major: top-to-bottom, left-to-right. Default: all False.
            button_config: List of (target_idx_or_idxs, action_type) per button.
                Parse order is row-major: top-to-bottom, left-to-right.
                target_idx_or_idxs may be a single int or a list/tuple of ints.
                Targets are moving wall indexes for moving-wall actions and
                barrier indexes for barrier actions. action_type is a
                ButtonAction enum value.
                Default: all (0, ButtonAction.TOGGLE_DIRECTION).
            barrier_config: List of bools per barrier. Parse order is row-major:
                top-to-bottom, left-to-right. Default: all False.

        Legacy:
            O: onion pile - will be interpreted as ingredient 0
        """
        rows = grid.strip("\n").split("\n")

        if len(rows[0]) == 0:
            rows = rows[1:]
        if len(rows[-1]) == 0:
            rows = rows[:-1]

        row_lens = [len(row) for row in rows]
        if strict_rectangular and len(set(row_lens)) != 1:
            raise ValueError(
                f"Layout rows must be rectangular, got row lengths: {row_lens}"
            )

        static_objects = np.zeros((len(rows), max(row_lens)), dtype=int)

        char_to_static_item = {
            " ": StaticObject.EMPTY,
            "W": StaticObject.WALL,
            "X": StaticObject.GOAL,
            "B": StaticObject.PLATE_PILE,
            "P": StaticObject.POT,
            "R": StaticObject.RECIPE_INDICATOR,
        }

        # Add ingredient piles 0-9
        for r in range(10):
            char_to_static_item[f"{r}"] = StaticObject.INGREDIENT_PILE_BASE + r

        # Item conveyor belt directions
        item_conveyor_chars = {
            ">": Direction.RIGHT,
            "<": Direction.LEFT,
            "^": Direction.UP,
            "v": Direction.DOWN,
        }

        # Player conveyor belt directions
        player_conveyor_chars = {
            "]": Direction.RIGHT,
            "[": Direction.LEFT,
            "{": Direction.UP,
            "}": Direction.DOWN,
        }
        valid_chars = (
            set(char_to_static_item)
            | set(item_conveyor_chars)
            | set(player_conveyor_chars)
            | {"A", "O"}
        )

        # Moving wall directions (compass: n=up, s=down, e=east/right, w=west/left)
        moving_wall_chars = {
            "n": Direction.UP,
            "s": Direction.DOWN,
            "e": Direction.RIGHT,
            "w": Direction.LEFT,
        }

        agent_positions = []
        item_conveyor_info = []
        player_conveyor_info = []
        moving_wall_positions = []  # (y, x, direction) before bounce applied
        button_positions = []       # (y, x)
        barrier_positions = []      # (y, x)

        num_ingredients = 0
        includes_recipe_indicator = False

        for r, row in enumerate(rows):
            c = 0
            while c < len(row):
                char = row[c]

                # Legacy: O -> 0 (onion)
                if char == "O":
                    char = "0"

                if char == "A":
                    agent_pos = (c, r)
                    agent_positions.append(agent_pos)
                    static_objects[r, c] = StaticObject.EMPTY
                elif char in item_conveyor_chars:
                    static_objects[r, c] = StaticObject.ITEM_CONVEYOR
                    direction = item_conveyor_chars[char]
                    item_conveyor_info.append((r, c, direction))
                elif char in player_conveyor_chars:
                    static_objects[r, c] = StaticObject.PLAYER_CONVEYOR
                    direction = player_conveyor_chars[char]
                    player_conveyor_info.append((r, c, direction))
                elif char in moving_wall_chars:
                    static_objects[r, c] = StaticObject.MOVING_WALL
                    direction = moving_wall_chars[char]
                    moving_wall_positions.append((r, c, direction))
                elif char == "!":
                    static_objects[r, c] = StaticObject.BUTTON
                    button_positions.append((r, c))
                elif char == "#":
                    static_objects[r, c] = StaticObject.BARRIER
                    barrier_positions.append((r, c))
                else:
                    if char not in valid_chars:
                        raise ValueError(
                            f"Unknown layout character {char!r} at row {r}, col {c}"
                        )
                    obj = char_to_static_item.get(char, StaticObject.EMPTY)
                    static_objects[r, c] = obj

                    if StaticObject.is_ingredient_pile(obj):
                        ingredient_idx = obj - StaticObject.INGREDIENT_PILE_BASE
                        num_ingredients = max(num_ingredients, ingredient_idx + 1)

                    if obj == StaticObject.RECIPE_INDICATOR:
                        includes_recipe_indicator = True

                c += 1

        if possible_recipes is None and not includes_recipe_indicator:
            raise ValueError(
                "Layout does not include a recipe indicator, a fixed recipe must be provided"
            )

        if swap_agents:
            agent_positions = agent_positions[::-1]

        # Ensure at least one ingredient type
        if num_ingredients == 0:
            num_ingredients = 1

        # Build moving wall info with bounce config
        if moving_wall_bounce is None:
            moving_wall_bounce = [False] * len(moving_wall_positions)
        if len(moving_wall_bounce) != len(moving_wall_positions):
            raise ValueError(
                f"moving_wall_bounce length ({len(moving_wall_bounce)}) must match "
                f"number of moving walls ({len(moving_wall_positions)})"
            )
        moving_wall_info = [
            (y, x, direction, bounce)
            for (y, x, direction), bounce in zip(moving_wall_positions, moving_wall_bounce)
        ]

        # Build button info with config
        if button_config is None:
            button_config = [(0, ButtonAction.TOGGLE_DIRECTION)] * len(button_positions)
        if len(button_config) != len(button_positions):
            raise ValueError(
                f"button_config length ({len(button_config)}) must match "
                f"number of buttons ({len(button_positions)})"
            )

        def _normalize_button_targets(target_idxs):
            if isinstance(target_idxs, (list, tuple)):
                return tuple(target_idxs)
            return (target_idxs,)

        button_info = [
            (y, x, _normalize_button_targets(target_idxs), action_type)
            for (y, x), (target_idxs, action_type) in zip(
                button_positions, button_config
            )
        ]

        # Build barrier info with config
        if barrier_config is None:
            barrier_config = [False] * len(barrier_positions)
        if len(barrier_config) != len(barrier_positions):
            raise ValueError(
                f"barrier_config length ({len(barrier_config)}) must match "
                f"number of barriers ({len(barrier_positions)})"
            )
        barrier_info = [
            (y, x, active)
            for (y, x), active in zip(barrier_positions, barrier_config)
        ]

        layout = Layout(
            agent_positions=agent_positions,
            static_objects=static_objects,
            num_ingredients=num_ingredients,
            possible_recipes=possible_recipes,
            item_conveyor_info=item_conveyor_info,
            player_conveyor_info=player_conveyor_info,
            moving_wall_info=moving_wall_info,
            button_info=button_info,
            barrier_info=barrier_info,
        )

        return layout


# Pre-defined layouts
overcooked_v3_layouts = {
    # Original Overcooked-AI layouts
    "cramped_room": Layout.from_string(
        cramped_room, possible_recipes=[[0, 0, 0]], swap_agents=True
    ),
    "asymm_advantages": Layout.from_string(
        asymm_advantages, possible_recipes=[[0, 0, 0]]
    ),
    "coord_ring": Layout.from_string(coord_ring, possible_recipes=[[0, 0, 0]]),
    "forced_coord": Layout.from_string(forced_coord, possible_recipes=[[0, 0, 0]]),
    "counter_circuit": Layout.from_string(
        counter_circuit, possible_recipes=[[0, 0, 0]], swap_agents=True
    ),

    # V2-style layouts with recipe indicators
    "cramped_room_v2": Layout.from_string(
        cramped_room_v2, possible_recipes=[[0, 0, 0]]
    ),

    # Demo layouts with conveyors
    "conveyor_demo": Layout.from_string(
        conveyor_demo, possible_recipes=[[0, 0, 0]]
    ),
    "player_conveyor_demo": Layout.from_string(
        player_conveyor_demo, possible_recipes=[[0, 0, 0]]
    ),

    # 2x2 clockwise conveyor loop for testing
    "player_conveyor_loop": Layout.from_string(
        player_conveyor_loop, possible_recipes=[[0, 0, 0]]
    ),

    # Moving wall demos
    "moving_wall_demo": Layout.from_string(
        moving_wall_demo,
        possible_recipes=[[0, 0, 0]],
        button_config=[(0, ButtonAction.TOGGLE_DIRECTION)],
    ),
    "moving_wall_bounce_demo": Layout.from_string(
        moving_wall_bounce_demo,
        possible_recipes=[[0, 0, 0]],
        moving_wall_bounce=[True, True],
        button_config=[(1, ButtonAction.TOGGLE_PAUSE)],
    ),

    # Barrier demo
    "barrier_demo": Layout.from_string(
        barrier_demo,
        possible_recipes=[[0, 0, 0]],
        barrier_config=[False, True],  # First barrier off, second barrier on initially
    ),

    # Timed barrier demo with button
    "timed_barrier_demo": Layout.from_string(
        timed_barrier_demo,
        possible_recipes=[[0, 0, 0]],
        barrier_config=[True, True],  # Barrier starts active
        button_config=[(0, ButtonAction.TIMED_BARRIER), (1, ButtonAction.TIMED_BARRIER)],  # Button controls barrier 0 with timed toggle
    ),
    "moving_wall_barrier_button_demo": Layout.from_string(
        moving_wall_barrier_button_demo,
        possible_recipes=[[0, 0, 0]],
        barrier_config=[True],
        button_config=[
            (0, ButtonAction.TOGGLE_DIRECTION),
            (0, ButtonAction.TOGGLE_BARRIER),
        ],
    ),
    "middle_conveyor": Layout.from_string(
        middle_conveyor, possible_recipes=[[0, 0, 0]],
    ),

    "follow_the_leader": Layout.from_string(
        follow_the_leader, possible_recipes=[[0, 0, 0]],
    ),

    "around_the_island": Layout.from_string(
        around_the_island, possible_recipes=[[0, 0, 0]],
    ),

    "single_file": Layout.from_string(
        single_file, possible_recipes=[[0, 0, 0]],
    ),

}
