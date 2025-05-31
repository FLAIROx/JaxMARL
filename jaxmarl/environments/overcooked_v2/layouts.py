from jaxmarl.environments.overcooked_v2.common import StaticObject
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import itertools

# Layouts from Overcooked-AI
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


# Adapted layouts
cramped_room_v2 = """
WWPWW
0A A1
W   R
WBWXW
"""
asymm_advantages_recipes_center = """
WWWWWWWWW
0 WXR01 X
1   P   W
W A PA  W
WWWBWBWWW
"""
asymm_advantages_recipes_right = """
WWWWWWWWW
0 WXW01 X
1   P   R
W A PA  W
WWWBWBWWW
"""
asymm_advantages_recipes_left = """
WWWWWWWWW
0 WXW01 X
1   P   R
R A PA  W
WWWBWBWWW
"""
two_rooms = """
WWWWWB10W
W   W   R
P A W A W
W   W   X
WWWWWWWWW
"""


# Other Layouts
two_rooms_both = """
W01BWB10W
W   W   R
P A W A W
W   W   X
WWWWWWWWW
"""
long_room = """
WWWWWWWWWWWWWWW
B            AP
0             X
WWWWWWWWWWWWWWW
"""
fun_coordination = """
WWWWWWWWW
0   X   2
RA  P  AW
1   B   3
WWWWWWWWW
"""
more_fun_coordination = """
WWWWWWWWW
W   X   W
RA  P  A1
0   P   2
W   B   W
WWWWWWWWW
"""
fun_symmetries_plates = """
WWWWWWW
B  W  0
R APA X
B  W  1
WWWWWWW
"""
fun_symmetries = """
WWWWBWW
2  W  0
R APA X
2  W  1
WWWWBWW
"""
fun_symmetries1 = """
WWWWWBWW
2  WW  0
R AWPA X
2  WW  1
WWWWWBWW
"""
overcookedv2_demo = """
WWPWW
0A A1
L   R
WBWXW
"""


# Extended Cat-Dog Problem Layouts
grounded_coord_simple = """
WW2WWWWW
W  WB  0
R ALPA X
W  WB  1
WW2WWWWW
"""
grounded_coord_ring = """
WWW2R2WWW
W       W
W WWLWW W
2 0   B 2
RAXAP X R
2 1   B 2
W WWLWW W
W       W
WWW2R2WWW
"""


# Test-Time Protocol Formation Layouts
test_time_simple = """
WW2WWWWW
W  WB  0
R AWPA X
W  WB  1
WW2WWWWW
"""
test_time_wide = """
WWXBWW
0 A  0
1    1
WPWPWW
3 A  3
W    W
WWRWWW
"""


# Demo Cook Layouts
demo_cook_simple = """
WWWWWR2W0WW
0      W  B
W     APA X
1      W  B
WWWWWR2W1WW
"""
demo_cook_wide = """
WWWWBXBWWWW
WWW0 A 1WWW
WWWWWPWWWWW
W    A    W
0  W3R3W  0
W1WWWWWWW1W
"""


@dataclass
class Layout:
    # agent positions list of positions, tuples (x, y)
    agent_positions: List[Tuple[int, int]]

    # width x height grid with static items
    static_objects: np.ndarray

    num_ingredients: int

    # If recipe is none, recipes will be sampled from the possible_recipes
    # If possible_recipes is none, all possible recipes with the available ingredients will be considered
    possible_recipes: Optional[List[List[int]]]

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

    @staticmethod
    def _get_all_possible_recipes(num_ingredients: int) -> List[List[int]]:
        """
        Get all possible recipes given the number of ingredients.
        """
        available_ingredients = list(range(num_ingredients)) * 3
        raw_combinations = itertools.combinations(available_ingredients, 3)
        unique_recipes = set(
            tuple(sorted(combination)) for combination in raw_combinations
        )

        return [list(recipe) for recipe in unique_recipes]

    @staticmethod
    def from_string(grid, possible_recipes=None, swap_agents=False):
        """Assumes `grid` is string representation of the layout, with 1 line per row, and the following symbols:
        W: wall
        A: agent
        X: goal
        B: plate (bowl) pile
        P: pot location
        R: recipe of the day indicator
        L: button recipe indicator
        0-9: Ingredient x pile
        ' ' (space) : empty cell

        Depricated:
        O: onion pile - will be interpreted as ingredient 0


        If `recipe` is provided, it should be a list of ingredient indices, max 3 ingredients per recipe
        If `recipe` is not provided, the recipe will be randomized on reset.
        If the layout does not have a recipe indicator, a fixed `recipe` must be provided.

        If `possible_recipes` is provided, it should be a list of lists of ingredient indices, 3 ingredients per recipe.

        Swap agents will swap the positions of the agents in the layout. This is only used for compatibility with the old Overcooked-AI layouts.
        """

        rows = grid.split("\n")

        if len(rows[0]) == 0:
            rows = rows[1:]
        if len(rows[-1]) == 0:
            rows = rows[:-1]

        row_lens = [len(row) for row in rows]
        static_objects = np.zeros((len(rows), max(row_lens)), dtype=int)

        char_to_static_item = {
            " ": StaticObject.EMPTY,
            "W": StaticObject.WALL,
            "X": StaticObject.GOAL,
            "B": StaticObject.PLATE_PILE,
            "P": StaticObject.POT,
            "R": StaticObject.RECIPE_INDICATOR,
            "L": StaticObject.BUTTON_RECIPE_INDICATOR,
        }

        for r in range(10):
            char_to_static_item[f"{r}"] = StaticObject.INGREDIENT_PILE_BASE + r

        agent_positions = []

        num_ingredients = 0
        includes_recipe_indicator = False
        for r, row in enumerate(rows):
            c = 0
            while c < len(row):
                char = row[c]

                if char == "O":
                    char = "0"

                if char == "A":
                    agent_pos = (c, r)
                    agent_positions.append(agent_pos)

                obj = char_to_static_item.get(char, StaticObject.EMPTY)
                static_objects[r, c] = obj

                if StaticObject.is_ingredient_pile(obj):
                    ingredient_idx = obj - StaticObject.INGREDIENT_PILE_BASE
                    num_ingredients = max(num_ingredients, ingredient_idx + 1)

                if obj == StaticObject.RECIPE_INDICATOR:
                    includes_recipe_indicator = True

                c += 1

        # TODO: add some sanity checks - e.g. agent must exist, surrounded by walls, etc.

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

        layout = Layout(
            agent_positions=agent_positions,
            static_objects=static_objects,
            num_ingredients=num_ingredients,
            possible_recipes=possible_recipes,
        )

        return layout


overcooked_v2_layouts = {
    # Overcooked-AI layouts
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
    # Adapted layouts
    "cramped_room_v2": Layout.from_string(cramped_room_v2),
    "asymm_advantages_recipes_center": Layout.from_string(
        asymm_advantages_recipes_center
    ),
    "asymm_advantages_recipes_right": Layout.from_string(
        asymm_advantages_recipes_right
    ),
    "asymm_advantages_recipes_left": Layout.from_string(asymm_advantages_recipes_left),
    "two_rooms": Layout.from_string(two_rooms),
    # Other layouts
    "two_rooms_both": Layout.from_string(two_rooms_both),
    "long_room": Layout.from_string(long_room, possible_recipes=[[0, 0, 0]]),
    "fun_coordination": Layout.from_string(
        fun_coordination, possible_recipes=[[0, 0, 2], [1, 1, 3]]
    ),
    "more_fun_coordination": Layout.from_string(
        more_fun_coordination, possible_recipes=[[0, 1, 1], [0, 2, 2]]
    ),
    "fun_symmetries": Layout.from_string(
        fun_symmetries, possible_recipes=[[0, 0, 0], [1, 1, 1]]
    ),
    "fun_symmetries_plates": Layout.from_string(
        fun_symmetries_plates, possible_recipes=[[0, 0, 0], [1, 1, 1]]
    ),
    "fun_symmetries1": Layout.from_string(
        fun_symmetries1, possible_recipes=[[0, 0, 0], [1, 1, 1]]
    ),
    # Extended Cat-Dog Problem Layouts
    "grounded_coord_simple": Layout.from_string(
        grounded_coord_simple, possible_recipes=[[0, 0, 0], [1, 1, 1]]
    ),
    "grounded_coord_ring": Layout.from_string(
        grounded_coord_ring, possible_recipes=[[0, 0, 0], [1, 1, 1]]
    ),
    # Test-Time Protocol Formation Layouts
    "test_time_simple": Layout.from_string(
        test_time_simple, possible_recipes=[[0, 0, 0], [1, 1, 1]]
    ),
    "test_time_wide": Layout.from_string(
        test_time_wide, possible_recipes=[[0, 0, 0], [1, 1, 1]]
    ),
    # Demo Cook Layouts
    "demo_cook_simple": Layout.from_string(
        demo_cook_simple, possible_recipes=[[0, 0, 0], [1, 1, 1]]
    ),
    "demo_cook_wide": Layout.from_string(
        demo_cook_wide, possible_recipes=[[0, 0, 0], [1, 1, 1]]
    ),
}
