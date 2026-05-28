"""Layout definitions and parsing for Overcooked V3.

DESIGN NOTES:
- don't make item conveyor belts / player conveyor belts move things to the same destination - this will cause race conditions and maybe make the items disappear.
"""

from jaxmarl.environments.overcooked_v3.common import StaticObject, Direction
from jaxmarl.environments.overcooked_v3.settings import MAX_POTS, MAX_ITEM_CONVEYORS, MAX_PLAYER_CONVEYORS
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

        item_conveyors = {(y, x): direction for y, x, direction in self.item_conveyor_info}
        player_conveyors = {(y, x): direction for y, x, direction in self.player_conveyor_info}

        for y in range(height):
            for x in range(width):
                obj = self.static_objects[y, x]

                if (y, x) in item_conveyors:
                    direction = Direction(item_conveyors[(y, x)])
                    grid[y][x] = item_conveyor_symbols[direction]
                elif (y, x) in player_conveyors:
                    direction = Direction(player_conveyors[(y, x)])
                    grid[y][x] = player_conveyor_symbols[direction]
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

        if self.possible_recipes is None or len(self.possible_recipes) == 0:
            if not info['has_recipe_indicator']:
                errors.append("Layout has no recipe indicator and no possible_recipes specified")
        else:
            for i, recipe in enumerate(self.possible_recipes):
                if not isinstance(recipe, list) or len(recipe) != 3:
                    errors.append(f"Recipe {i} must be a list of exactly 3 ingredient indices")
                else:
                    for ingredient_idx in recipe:
                        if ingredient_idx not in info['num_ingredient_piles'] and ingredient_idx < self.num_ingredients:
                            warnings.append(f"Recipe uses ingredient {ingredient_idx} but no pile exists in layout")

        all_messages = errors + warnings
        is_valid = len(errors) == 0

        return is_valid, all_messages

    @staticmethod
    def _is_agent_walkable_tile(obj) -> bool:
        return obj in (StaticObject.EMPTY, StaticObject.PLAYER_CONVEYOR)

    def _has_adjacent_walkable_tile(self, y: int, x: int) -> bool:
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            adj_y = y + dy
            adj_x = x + dx
            if not (0 <= adj_y < self.height and 0 <= adj_x < self.width):
                continue
            if self._is_agent_walkable_tile(self.static_objects[adj_y, adj_x]):
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
        grid, possible_recipes=None, swap_agents=False, strict_rectangular=True
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

        Legacy:
            O: onion pile - will be interpreted as ingredient 0
        """
        rows = grid.split("\n")

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

        agent_positions = []
        item_conveyor_info = []
        player_conveyor_info = []

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

        # Validation for recipes 
        # NOTE: possible_recipes is a list of lists of ingredient indices, max 3 ingredients per recipe - if no recipe indicator, we just auto-gen all possible combinations. Otherwise we just take the possible_recipes specified in layout. 
        if possible_recipes is not None:
            if not isinstance(possible_recipes, list):
                raise ValueError("possible_recipes must be a list")
            if not all(isinstance(recipe, list) for recipe in possible_recipes):
                raise ValueError("possible_recipes must be a list of lists")
            if not all(len(recipe) == 3 for recipe in possible_recipes):
                raise ValueError("All recipes must be of length 3")
        elif not includes_recipe_indicator:
            raise ValueError(
                "Layout does not include a recipe indicator, a fixed recipe must be provided"
            )

        if swap_agents:
            agent_positions = agent_positions[::-1]

        # Ensure at least one ingredient type
        if num_ingredients == 0:
            num_ingredients = 1

        layout = Layout(
            agent_positions=agent_positions,
            static_objects=static_objects,
            num_ingredients=num_ingredients,
            possible_recipes=possible_recipes,
            item_conveyor_info=item_conveyor_info,
            player_conveyor_info=player_conveyor_info,
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
