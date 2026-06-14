#!/usr/bin/env python3
"""Interactive visual level editor for Overcooked V3.

Controls:
  Left Click: Place selected object
  Right Click: Erase object

  Keyboard shortcuts:
    W = Wall          P = Pot           B = Plate Pile
    X = Delivery      A = Agent         R = Recipe Indicator
    0-9 = Ingredients E = Erase

    Ctrl+Z = Undo     Ctrl+Y = Redo
    Ctrl+N = New      Ctrl+O = Open     Ctrl+S = Save
    Ctrl+E = Export   Ctrl+T = Test Play

  Menu clicks:
    New, Load, Export, Test, Validate, Quit
"""

import sys
import json
import pygame
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from pathlib import Path

# Lazy-load JaxMARL dependencies to avoid requiring full jaxmarl installation
DEPENDENCIES_AVAILABLE = False
StaticObject = None
Direction = None
Layout = None
overcooked_v3_layouts = None
OvercookedV3Visualizer = None

# Fallback static object IDs (match Overcooked V3 StaticObject values)
FALLBACK_WALL = 1
FALLBACK_GOAL = 4
FALLBACK_POT = 5
FALLBACK_RECIPE = 6
FALLBACK_PLATE_PILE = 9
FALLBACK_INGREDIENT_BASE = 10
FALLBACK_ITEM_CONVEYOR = 20
FALLBACK_PLAYER_CONVEYOR = 21


def _load_jaxmarl_deps():
    """Load JaxMARL dependencies on first use."""
    global DEPENDENCIES_AVAILABLE, StaticObject, Direction, Layout, overcooked_v3_layouts
    global OvercookedV3Visualizer


    if DEPENDENCIES_AVAILABLE:
        return

    try:
        from jaxmarl.environments.overcooked_v3.common import (
            StaticObject as SO,
            Direction as Dir,
        )
        from jaxmarl.environments.overcooked_v3.layouts import (
            Layout as L,
            overcooked_v3_layouts as layouts,
        )
        from jaxmarl.viz.overcooked_v3_visualizer import OvercookedV3Visualizer as Viz

        StaticObject = SO
        Direction = Dir
        Layout = L
        overcooked_v3_layouts = layouts
        OvercookedV3Visualizer = Viz
        DEPENDENCIES_AVAILABLE = True
    except ImportError as e:
        print(f"Warning: Could not import JaxMARL dependencies: {e}")
        print("Editor will run in limited mode")
        DEPENDENCIES_AVAILABLE = False


# Colors (matching game visuals)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_GRAY = (100, 100, 100)
COLOR_LIGHT_GRAY = (180, 180, 180)
COLOR_DARK_GRAY = (64, 64, 64)
COLOR_RED = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (0, 100, 255)
COLOR_YELLOW = (255, 255, 0)
COLOR_ORANGE = (255, 165, 0)
COLOR_PURPLE = (160, 32, 240)
COLOR_CYAN = (0, 255, 255)
COLOR_BROWN = (139, 69, 19)
COLOR_DARK_GREEN = (0, 150, 0)
COLOR_PINK = (255, 105, 180)

# Ingredient colors
INGREDIENT_COLORS = [
    COLOR_YELLOW,  # Onion (0)
    COLOR_RED,  # Tomato (1)
    COLOR_DARK_GREEN,  # Lettuce (2)
    COLOR_CYAN,  # (3)
    COLOR_ORANGE,  # (4)
    COLOR_PURPLE,  # (5)
    COLOR_BLUE,  # (6)
    COLOR_PINK,  # (7)
    COLOR_BROWN,  # (8)
    COLOR_WHITE,  # (9)
]

# Agent colors
AGENT_COLORS = [
    COLOR_BLUE,
    COLOR_GREEN,
    COLOR_RED,
    COLOR_PURPLE,
    COLOR_YELLOW,
    COLOR_ORANGE,
]

# Layout
TILE_SIZE = 48
TOOLBAR_WIDTH = 250
INFO_PANEL_WIDTH = 300
TOP_MENU_HEIGHT = 40
MIN_GRID_WIDTH = 5
MIN_GRID_HEIGHT = 4
MAX_GRID_WIDTH = 20
MAX_GRID_HEIGHT = 20
DEFAULT_GRID_WIDTH = 7
DEFAULT_GRID_HEIGHT = 5


@dataclass
class EditorTool:
    """Represents a tool/object that can be placed."""

    name: str
    symbol: str
    description: str
    color: Tuple[int, int, int]
    static_object: Optional[int] = None
    conveyor_direction: Optional[int] = None
    is_agent: bool = False
    is_erase: bool = False
    keyboard_shortcut: Optional[str] = None


# Define all available tools
TOOLS = [
    EditorTool(
        "Wall",
        "W",
        "Blocks movement and acts as a counter.",
        COLOR_GRAY,
        StaticObject.WALL if DEPENDENCIES_AVAILABLE else FALLBACK_WALL,
        keyboard_shortcut="w",
    ),
    EditorTool(
        "Pot",
        "P",
        "Cooks ingredients into dishes.",
        COLOR_ORANGE,
        StaticObject.POT if DEPENDENCIES_AVAILABLE else FALLBACK_POT,
        keyboard_shortcut="p",
    ),
    EditorTool(
        "Plate Pile",
        "B",
        "Provides clean plates for serving.",
        COLOR_WHITE,
        StaticObject.PLATE_PILE if DEPENDENCIES_AVAILABLE else FALLBACK_PLATE_PILE,
        keyboard_shortcut="b",
    ),
    EditorTool(
        "Delivery",
        "X",
        "Deliver completed dishes here.",
        COLOR_GREEN,
        StaticObject.GOAL if DEPENDENCIES_AVAILABLE else FALLBACK_GOAL,
        keyboard_shortcut="x",
    ),
    EditorTool(
        "Recipe",
        "R",
        "Shows the current recipe to cook.",
        COLOR_PURPLE,
        StaticObject.RECIPE_INDICATOR if DEPENDENCIES_AVAILABLE else FALLBACK_RECIPE,
        keyboard_shortcut="r",
    ),
    EditorTool(
        "Agent",
        "A",
        "Sets an agent spawn position.",
        COLOR_BLUE,
        is_agent=True,
        keyboard_shortcut="a",
    ),
    EditorTool(
        "Ingredient 0",
        "0",
        "Onion pile (ingredient source).",
        COLOR_YELLOW,
        StaticObject.INGREDIENT_PILE_BASE
        if DEPENDENCIES_AVAILABLE
        else FALLBACK_INGREDIENT_BASE,
        keyboard_shortcut="0",
    ),
    EditorTool(
        "Ingredient 1",
        "1",
        "Tomato pile (ingredient source).",
        COLOR_RED,
        (StaticObject.INGREDIENT_PILE_BASE + 1)
        if DEPENDENCIES_AVAILABLE
        else (FALLBACK_INGREDIENT_BASE + 1),
        keyboard_shortcut="1",
    ),
    EditorTool(
        "Ingredient 2",
        "2",
        "Lettuce pile (ingredient source).",
        (0, 150, 0),
        (StaticObject.INGREDIENT_PILE_BASE + 2)
        if DEPENDENCIES_AVAILABLE
        else (FALLBACK_INGREDIENT_BASE + 2),
        keyboard_shortcut="2",
    ),
    EditorTool(
        "Item Conv >",
        ">",
        "Moves items to the right.",
        COLOR_CYAN,
        conveyor_direction=2,
        keyboard_shortcut=">",
    ),  # Direction.RIGHT
    EditorTool(
        "Item Conv <",
        "<",
        "Moves items to the left.",
        COLOR_CYAN,
        conveyor_direction=3,
        keyboard_shortcut="<",
    ),  # Direction.LEFT
    EditorTool(
        "Item Conv ^",
        "^",
        "Moves items upward.",
        COLOR_CYAN,
        conveyor_direction=0,
        keyboard_shortcut="^",
    ),  # Direction.UP
    EditorTool(
        "Item Conv v",
        "v",
        "Moves items downward.",
        COLOR_CYAN,
        conveyor_direction=1,
        keyboard_shortcut="v",
    ),  # Direction.DOWN
    EditorTool(
        "Player Conv ]",
        "]",
        "Moves agents to the right.",
        COLOR_PURPLE,
        conveyor_direction=2,
        keyboard_shortcut="]",
    ),  # Direction.RIGHT
    EditorTool(
        "Player Conv [",
        "[",
        "Moves agents to the left.",
        COLOR_PURPLE,
        conveyor_direction=3,
        keyboard_shortcut="[",
    ),  # Direction.LEFT
    EditorTool(
        "Player Conv {",
        "{",
        "Moves agents upward.",
        COLOR_PURPLE,
        conveyor_direction=0,
        keyboard_shortcut="{",
    ),  # Direction.UP
    EditorTool(
        "Player Conv }",
        "}",
        "Moves agents downward.",
        COLOR_PURPLE,
        conveyor_direction=1,
        keyboard_shortcut="}",
    ),  # Direction.DOWN
    EditorTool(
        "Erase",
        "⌫",
        "Removes objects from a tile.",
        COLOR_RED,
        is_erase=True,
        keyboard_shortcut="e",
    ),
]

# Create shortcut lookup
SHORTCUT_TO_TOOL = {
    tool.keyboard_shortcut: i for i, tool in enumerate(TOOLS) if tool.keyboard_shortcut
}


@dataclass
class EditorState:
    """Current state of the level editor."""

    width: int = DEFAULT_GRID_WIDTH
    height: int = DEFAULT_GRID_HEIGHT
    static_objects: np.ndarray = field(
        default_factory=lambda: np.zeros(
            (DEFAULT_GRID_HEIGHT, DEFAULT_GRID_WIDTH), dtype=int
        )
    )
    agent_positions: List[Tuple[int, int]] = field(default_factory=list)
    item_conveyors: Dict[Tuple[int, int], int] = field(
        default_factory=dict
    )  # (y,x) -> direction
    player_conveyors: Dict[Tuple[int, int], int] = field(
        default_factory=dict
    )  # (y,x) -> direction
    selected_tool: int = 0  # Index into TOOLS
    undo_stack: List[dict] = field(default_factory=list)
    redo_stack: List[dict] = field(default_factory=list)
    layout_name: str = "custom_layout"
    recipes: List[List[int]] = field(default_factory=lambda: [[0, 0, 0]])
    modified: bool = False

    def clone(self) -> dict:
        """Create a snapshot of current state for undo/redo."""
        return {
            "static_objects": self.static_objects.copy(),
            "agent_positions": self.agent_positions.copy(),
            "item_conveyors": self.item_conveyors.copy(),
            "player_conveyors": self.player_conveyors.copy(),
        }

    def restore(self, snapshot: dict):
        """Restore state from snapshot."""
        self.static_objects = snapshot["static_objects"]
        self.agent_positions = snapshot["agent_positions"]
        self.item_conveyors = snapshot["item_conveyors"]
        self.player_conveyors = snapshot["player_conveyors"]

    def save_undo(self):
        """Save current state to undo stack."""
        self.undo_stack.append(self.clone())
        self.redo_stack.clear()
        # Limit undo stack size
        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0)

    def undo(self):
        """Undo last action."""
        if self.undo_stack:
            self.redo_stack.append(self.clone())
            self.restore(self.undo_stack.pop())
            self.modified = True

    def redo(self):
        """Redo last undone action."""
        if self.redo_stack:
            self.undo_stack.append(self.clone())
            self.restore(self.redo_stack.pop())
            self.modified = True

    def resize(self, new_width: int, new_height: int):
        """Resize the grid, preserving existing content where possible."""
        self.save_undo()

        new_grid = np.zeros((new_height, new_width), dtype=int)

        # Copy old content
        copy_h = min(self.height, new_height)
        copy_w = min(self.width, new_width)
        new_grid[:copy_h, :copy_w] = self.static_objects[:copy_h, :copy_w]

        # Remove agents/conveyors that are now out of bounds
        self.agent_positions = [
            (x, y) for x, y in self.agent_positions if x < new_width and y < new_height
        ]
        self.item_conveyors = {
            (y, x): d
            for (y, x), d in self.item_conveyors.items()
            if x < new_width and y < new_height
        }
        self.player_conveyors = {
            (y, x): d
            for (y, x), d in self.player_conveyors.items()
            if x < new_width and y < new_height
        }

        self.static_objects = new_grid
        self.width = new_width
        self.height = new_height
        self.modified = True

    def clear(self):
        """Clear the entire grid."""
        self.save_undo()
        self.static_objects = np.zeros((self.height, self.width), dtype=int)
        self.agent_positions.clear()
        self.item_conveyors.clear()
        self.player_conveyors.clear()
        self.modified = True

    def to_layout(self) -> "Layout":
        """Convert editor state to a Layout object."""
        if not DEPENDENCIES_AVAILABLE:
            raise RuntimeError("Cannot create Layout without JaxMARL dependencies")

        # Convert item_conveyors dict to list of tuples
        item_conveyor_info = [(y, x, d) for (y, x), d in self.item_conveyors.items()]
        player_conveyor_info = [
            (y, x, d) for (y, x), d in self.player_conveyors.items()
        ]

        # Determine number of ingredients
        num_ingredients = 1
        for obj in self.static_objects.flat:
            if StaticObject.is_ingredient_pile(obj):
                ingredient_idx = obj - StaticObject.INGREDIENT_PILE_BASE
                num_ingredients = max(num_ingredients, ingredient_idx + 1)

        return Layout(
            agent_positions=self.agent_positions,
            static_objects=self.static_objects.copy(),
            num_ingredients=num_ingredients,
            possible_recipes=self.recipes if self.recipes else None,
            item_conveyor_info=item_conveyor_info,
            player_conveyor_info=player_conveyor_info,
        )


class LevelEditor:
    """Main level editor application."""

    def __init__(self):
        pygame.init()

        # Calculate window size (50/50 split between grid and info panel)
        self.tile_size = TILE_SIZE
        self.grid_width = DEFAULT_GRID_WIDTH * self.tile_size
        self.grid_height = DEFAULT_GRID_HEIGHT * self.tile_size
        self.window_width = TOOLBAR_WIDTH + self.grid_width * 2
        self.window_height = TOP_MENU_HEIGHT + self.grid_height

        self.screen = pygame.display.set_mode(
            (self.window_width, self.window_height), pygame.RESIZABLE
        )
        pygame.display.set_caption("Overcooked V3 Level Editor")

        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        self.title_font = pygame.font.Font(None, 32)

        self.state = EditorState()
        self.running = True
        self.hover_pos = None

        # UI state
        self.show_export_dialog = False
        self.export_text = ""
        self.show_load_dialog = False
        self.selected_layout_name = None
        self.validation_messages = []
        self.test_play_process = None

        # Resize dialog state
        self.show_resize_dialog = False
        self.resize_width_input = str(DEFAULT_GRID_WIDTH)
        self.resize_height_input = str(DEFAULT_GRID_HEIGHT)
        self.resize_input_mode = "width"  # 'width' or 'height'
        self.resize_error = ""

        # Paint-drag state
        self.painting = False
        self.paint_button = None  # 1 = left (place), 3 = right (erase)
        self.last_paint_cell = None  # (x, y) to avoid re-painting same cell

    def run(self):
        """Main editor loop."""
        while self.running:
            self.handle_events()
            self.draw()
            self.clock.tick(60)

        pygame.quit()

    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.handle_mouse_click(event.button, event.pos)

            elif event.type == pygame.MOUSEBUTTONUP:
                self.painting = False
                self.paint_button = None
                self.last_paint_cell = None

            elif event.type == pygame.MOUSEMOTION:
                self.hover_pos = event.pos
                if self.painting:
                    self._handle_paint_drag(event.pos)

            elif event.type == pygame.KEYDOWN:
                self.handle_keypress(event)

            elif event.type == pygame.VIDEORESIZE:
                self._set_window_size(event.w, event.h)

    def _set_window_size(self, width: int, height: int):
        """Update the window size while keeping the window resizable."""
        self.window_width = width
        self.window_height = height
        self._recalc_tile_size()
        self.screen = pygame.display.set_mode(
            (self.window_width, self.window_height), pygame.RESIZABLE
        )

    def _recalc_tile_size(self):
        """Recalculate tile size so the grid fills 50% of the space after the toolbar."""
        available = self.window_width - TOOLBAR_WIDTH
        half = available // 2
        self.tile_size = max(16, half // self.state.width)
        self.grid_width = self.state.width * self.tile_size
        self.grid_height = self.state.height * self.tile_size

    def handle_mouse_click(self, button: int, pos: Tuple[int, int]):
        """Handle mouse clicks."""
        mx, my = pos

        # Check if click is in toolbar
        if mx < TOOLBAR_WIDTH:
            self.handle_toolbar_click(mx, my)
            return

        # Check if click is in top menu
        if my < TOP_MENU_HEIGHT:
            self.handle_menu_click(mx, my)
            return

        # Check if click is in grid
        grid_x = mx - TOOLBAR_WIDTH
        grid_y = my - TOP_MENU_HEIGHT

        if (
            0 <= grid_x < self.state.width * self.tile_size
            and 0 <= grid_y < self.state.height * self.tile_size
        ):
            cell_x = grid_x // self.tile_size
            cell_y = grid_y // self.tile_size

            if button == 1:  # Left click - place
                self.place_object(cell_x, cell_y)
                # Start drag-painting
                self.painting = True
                self.paint_button = 1
                self.last_paint_cell = (cell_x, cell_y)
            elif button == 3:  # Right click - erase
                self.erase_object(cell_x, cell_y)
                # Start drag-erasing
                self.painting = True
                self.paint_button = 3
                self.last_paint_cell = (cell_x, cell_y)

    def _handle_paint_drag(self, pos: Tuple[int, int]):
        """Handle painting while dragging the mouse."""
        mx, my = pos
        grid_x = mx - TOOLBAR_WIDTH
        grid_y = my - TOP_MENU_HEIGHT

        if (
            0 <= grid_x < self.state.width * self.tile_size
            and 0 <= grid_y < self.state.height * self.tile_size
        ):
            cell_x = grid_x // self.tile_size
            cell_y = grid_y // self.tile_size

            # Skip if same cell as last paint
            if (cell_x, cell_y) == self.last_paint_cell:
                return

            self.last_paint_cell = (cell_x, cell_y)

            if self.paint_button == 1:  # Left drag - place
                # Skip agents during drag (don't spawn many agents by accident)
                tool = TOOLS[self.state.selected_tool]
                if tool.is_agent:
                    return
                self.place_object(cell_x, cell_y, save_undo=False)
            elif self.paint_button == 3:  # Right drag - erase
                self.erase_object(cell_x, cell_y, save_undo=False)
        else:
            # Cursor left the grid, stop painting
            self.painting = False
            self.paint_button = None
            self.last_paint_cell = None

    def handle_toolbar_click(self, mx: int, my: int):
        """Handle clicks in the toolbar."""
        # Skip top menu area
        toolbar_y = my - TOP_MENU_HEIGHT
        if toolbar_y < 0:
            return

        # Calculate which tool was clicked
        tool_height = 35
        tool_idx = toolbar_y // tool_height

        if 0 <= tool_idx < len(TOOLS):
            self.state.selected_tool = tool_idx

    def handle_menu_click(self, mx: int, my: int):
        """Handle clicks in the top menu bar."""
        menu_items = ["New", "Load", "Export", "Resize", "Test", "Quit"]
        item_width = 80

        item_idx = (mx - TOOLBAR_WIDTH) // item_width
        if 0 <= item_idx < len(menu_items):
            action = menu_items[item_idx]

            if action == "New":
                self.new_layout()
            elif action == "Load":
                self.show_load_dialog = True
            elif action == "Export":
                self.export_layout()
            elif action == "Resize":
                self.show_resize_dialog = True
                self.resize_error = ""
                self.resize_width_input = str(self.state.width)
                self.resize_height_input = str(self.state.height)
            elif action == "Test":
                self.test_play()
            elif action == "Quit":
                self.running = False

    def handle_keypress(self, event):
        """Handle keyboard shortcuts."""
        key = pygame.key.name(event.key)
        mods = pygame.key.get_mods()

        # Handle text input for resize dialog
        if self.show_resize_dialog:
            if key == "backspace":
                if self.resize_input_mode == "width":
                    self.resize_width_input = self.resize_width_input[:-1]
                else:
                    self.resize_height_input = self.resize_height_input[:-1]
            elif key == "tab":
                # Switch input focus
                self.resize_input_mode = (
                    "height" if self.resize_input_mode == "width" else "width"
                )
            elif key == "return":
                # Try to apply resize
                try:
                    new_width = int(self.resize_width_input)
                    new_height = int(self.resize_height_input)

                    if (
                        MIN_GRID_WIDTH <= new_width <= MAX_GRID_WIDTH
                        and MIN_GRID_HEIGHT <= new_height <= MAX_GRID_HEIGHT
                    ):
                        self.state.resize(new_width, new_height)
                        self.show_resize_dialog = False

                        # Recalc tile size for current window
                        self._recalc_tile_size()
                except ValueError:
                    pass
            elif key in "0123456789":
                # Add digit to current input
                if self.resize_input_mode == "width":
                    if len(self.resize_width_input) < 2:
                        self.resize_width_input += key
                else:
                    if len(self.resize_height_input) < 2:
                        self.resize_height_input += key
            elif key == "escape":
                self.show_resize_dialog = False
            return

        # Check for tool shortcuts (use event.unicode for shifted chars like >, <, {, }, ^)
        char = event.unicode
        if char in SHORTCUT_TO_TOOL:
            self.state.selected_tool = SHORTCUT_TO_TOOL[char]
            return
        if key in SHORTCUT_TO_TOOL:
            self.state.selected_tool = SHORTCUT_TO_TOOL[key]
            return

        # Check for ctrl combinations
        if mods & pygame.KMOD_CTRL:
            if key == "z":
                self.state.undo()
            elif key == "y":
                self.state.redo()
            elif key == "n":
                self.new_layout()
            elif key == "e":
                self.export_layout()
            elif key == "t":
                self.test_play()

    def place_object(self, x: int, y: int, save_undo: bool = True):
        """Place the selected object at the given grid position."""
        if save_undo:
            self.state.save_undo()

        tool = TOOLS[self.state.selected_tool]

        # Remove anything at this position first
        self.erase_object(x, y, save_undo=False)

        if tool.is_agent:
            self.state.agent_positions.append((x, y))
        elif tool.is_erase:
            pass  # Already erased
        elif tool.conveyor_direction is not None:
            # Determine if item or player conveyor based on tool name
            if "Player" in tool.name:
                self.state.player_conveyors[(y, x)] = tool.conveyor_direction
                if DEPENDENCIES_AVAILABLE:
                    self.state.static_objects[y, x] = StaticObject.PLAYER_CONVEYOR
                else:
                    self.state.static_objects[y, x] = FALLBACK_PLAYER_CONVEYOR
            else:
                self.state.item_conveyors[(y, x)] = tool.conveyor_direction
                if DEPENDENCIES_AVAILABLE:
                    self.state.static_objects[y, x] = StaticObject.ITEM_CONVEYOR
                else:
                    self.state.static_objects[y, x] = FALLBACK_ITEM_CONVEYOR
        elif tool.static_object is not None:
            self.state.static_objects[y, x] = tool.static_object

        self.state.modified = True

    def erase_object(self, x: int, y: int, save_undo: bool = True):
        """Erase object at the given position."""
        if save_undo:
            self.state.save_undo()

        # Remove agent if present
        self.state.agent_positions = [
            (ax, ay)
            for ax, ay in self.state.agent_positions
            if not (ax == x and ay == y)
        ]

        # Remove conveyors
        if (y, x) in self.state.item_conveyors:
            del self.state.item_conveyors[(y, x)]
        if (y, x) in self.state.player_conveyors:
            del self.state.player_conveyors[(y, x)]

        # Clear static object
        self.state.static_objects[y, x] = 0

        if save_undo:
            self.state.modified = True

    def new_layout(self):
        """Create a new empty layout."""
        # TODO: Add confirmation dialog if modified
        self.state = EditorState()
        self.validation_messages = []
        self.show_load_dialog = False
        self.show_export_dialog = False

    def export_layout(self):
        """Export the current layout as code."""
        try:
            layout_str = self._layout_string_from_state()

            # Generate code snippet
            code = f'''# Add to jaxmarl/environments/overcooked_v3/layouts.py

{self.state.layout_name} = """
{layout_str.strip()}
"""

# Add to overcooked_v3_layouts dictionary:
overcooked_v3_layouts["{self.state.layout_name}"] = Layout.from_string(
    {self.state.layout_name},
    possible_recipes={self.state.recipes},
    swap_agents=False
)
'''
            export_dir = Path(__file__).resolve().parents[2] / "exports"
            export_dir.mkdir(parents=True, exist_ok=True)
            export_path = export_dir / f"{self.state.layout_name}.txt"
            export_path.write_text(code, encoding="utf-8")

            self.export_text = str(export_path)
            self.show_export_dialog = True
            print("\n" + "=" * 60)
            print("EXPORTED LAYOUT CODE:")
            print("=" * 60)
            print(code)
            print(f"Saved to: {export_path}")
            print("=" * 60)

        except Exception as e:
            print(f"Error exporting layout: {e}")
            self.validation_messages = [f"Export error: {e}"]

    def _layout_string_from_state(self) -> str:
        """Build a Layout.from_string-compatible layout string from editor state."""
        height, width = self.state.height, self.state.width
        grid = [[" " for _ in range(width)] for _ in range(height)]

        if DEPENDENCIES_AVAILABLE and StaticObject is not None:
            ingredient_base = StaticObject.INGREDIENT_PILE_BASE
            item_conveyor_base = StaticObject.ITEM_CONVEYOR
            static_to_symbol = {
                StaticObject.WALL: "W",
                StaticObject.GOAL: "X",
                StaticObject.PLATE_PILE: "B",
                StaticObject.POT: "P",
                StaticObject.RECIPE_INDICATOR: "R",
            }
        else:
            ingredient_base = FALLBACK_INGREDIENT_BASE
            item_conveyor_base = FALLBACK_ITEM_CONVEYOR
            static_to_symbol = {
                FALLBACK_WALL: "W",
                FALLBACK_GOAL: "X",
                FALLBACK_PLATE_PILE: "B",
                FALLBACK_POT: "P",
                FALLBACK_RECIPE: "R",
            }

        item_symbols = {
            2: ">",
            3: "<",
            0: "^",
            1: "v",
        }  # RIGHT: >, LEFT: <, UP: ^, DOWN: v
        player_symbols = {
            2: "]",
            3: "[",
            0: "{",
            1: "}",
        }  # RIGHT: ], LEFT: [, UP: {, DOWN: }

        for y in range(height):
            for x in range(width):
                if (y, x) in self.state.item_conveyors:
                    direction = self.state.item_conveyors[(y, x)]
                    grid[y][x] = item_symbols.get(direction, ">")
                    continue
                if (y, x) in self.state.player_conveyors:
                    direction = self.state.player_conveyors[(y, x)]
                    grid[y][x] = player_symbols.get(direction, "]")
                    continue

                obj = self.state.static_objects[y, x]
                if obj in static_to_symbol:
                    grid[y][x] = static_to_symbol[obj]
                elif ingredient_base <= obj < item_conveyor_base:
                    grid[y][x] = str(obj - ingredient_base)

        for agent_x, agent_y in self.state.agent_positions:
            if 0 <= agent_x < width and 0 <= agent_y < height:
                grid[agent_y][agent_x] = "A"

        lines = ["".join(row) for row in grid]
        return "\n" + "\n".join(lines) + "\n"

    def test_play(self):
        """Launch test play mode with current layout."""
        if not DEPENDENCIES_AVAILABLE:
            print("Cannot test play: JaxMARL dependencies not available")
            return

        try:
            # First validate the layout
            layout = self.state.to_layout()
            is_valid, messages = layout.validate()


            if not is_valid:
                self.validation_messages = [
                    "Cannot test - layout has errors:"
                ] + messages
                print("\nCannot test play - please fix validation errors first")
                for msg in messages:
                    print(f"  - {msg}")
                return

            # Save layout temporarily
            import tempfile
            import subprocess

            layout_str = layout.to_string()
            recipes_str = str(self.state.recipes)

            # Create a temporary test script
            test_script = f'''#!/usr/bin/env python3
import jax
import pygame
import numpy as np
from jaxmarl import make
from jaxmarl.environments.overcooked_v3.layouts import Layout
from jaxmarl.viz.overcooked_v3_visualizer import OvercookedV3Visualizer

# Test layout
layout_str = """{layout_str}"""

layout = Layout.from_string(layout_str, possible_recipes={recipes_str})

env = make('overcooked_v3')
env.layout = layout

print("\\n" + "="*60)
print("TEST PLAY MODE - Press Q to quit and return to editor")
print("="*60)
print("Agent 0: WASD + Space")
print("Agent 1: Arrows + Enter")
print("="*60 + "\\n")

viz = OvercookedV3Visualizer(env, tile_size=48)
pygame.init()

width = env.width * 48
height = env.height * 48
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Test Play - Press Q to quit")
clock = pygame.time.Clock()

key = jax.random.PRNGKey(42)
key, subkey = jax.random.split(key)
obs, state = env.reset(subkey)

AGENT0_KEYS = {{
    pygame.K_w: 3, pygame.K_s: 1, pygame.K_a: 2, pygame.K_d: 0, pygame.K_SPACE: 5,
}}

AGENT1_KEYS = {{
    pygame.K_UP: 3, pygame.K_DOWN: 1, pygame.K_LEFT: 2, pygame.K_RIGHT: 0, pygame.K_RETURN: 5,
}}

running = True
total_reward = 0

while running:
    agent0_action = 4
    agent1_action = 4
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_r:
                key, subkey = jax.random.split(key)
                obs, state = env.reset(subkey)
                total_reward = 0
    
    keys = pygame.key.get_pressed()
    
    for k, action in AGENT0_KEYS.items():
        if keys[k]:
            agent0_action = action
            break
    
    for k, action in AGENT1_KEYS.items():
        if keys[k]:
            agent1_action = action
            break
    
    actions = {{'agent_0': agent0_action, 'agent_1': agent1_action}}
    key, subkey = jax.random.split(key)
    obs, state, rewards, dones, info = env.step(subkey, state, actions)
    
    total_reward += rewards['agent_0']
    
    img = viz.render_state(state)
    img_np = np.array(img)
    surf = pygame.surfarray.make_surface(img_np.swapaxes(0, 1))
    screen.blit(surf, (0, 0))
    
    font = pygame.font.Font(None, 24)
    hud_text = f"Score: {{total_reward:.0f}}  (Press Q to quit)"
    text_surf = font.render(hud_text, True, (255, 255, 255))
    screen.blit(text_surf, (5, 5))
    
    pygame.display.flip()
    clock.tick(10)

pygame.quit()
print("\\nReturning to editor...")
'''

            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(test_script)
                temp_file = f.name

            print(f"\nLaunching test play mode...")
            print(f"Temporary script: {temp_file}")

            # Run the test script (blocking)
            subprocess.run(["python3", temp_file])

            # Clean up
            import os

            os.unlink(temp_file)

        except Exception as e:
            print(f"Error launching test play: {e}")
            import traceback

            traceback.print_exc()
            self.validation_messages = [f"Test play error: {e}"]

    def validate(self):
        """Validate the current layout."""
        if not DEPENDENCIES_AVAILABLE:
            self.validation_messages = ["Cannot validate: dependencies not available"]
            return

        try:
            layout = self.state.to_layout()
            is_valid, messages = layout.validate()
            self.validation_messages = messages if messages else ["✓ Layout is valid!"]

            print("\n" + "=" * 60)
            print("VALIDATION RESULTS:")
            print("=" * 60)
            for msg in self.validation_messages:
                print(f"  {msg}")
            print("=" * 60 + "\n")

        except Exception as e:
            self.validation_messages = [f"Validation error: {e}"]

    def draw(self):
        """Draw the entire editor interface."""
        self.screen.fill(COLOR_BLACK)

        # Draw three panels
        self.draw_toolbar()
        self.draw_grid()
        self.draw_info_panel()
        self.draw_menu_bar()

        # Draw dialogs if active
        if self.show_export_dialog:
            self.draw_export_dialog()
        elif self.show_load_dialog:
            self.draw_load_dialog()
        elif self.show_resize_dialog:
            self.draw_resize_dialog()

        pygame.display.flip()

    def draw_menu_bar(self):
        """Draw the top menu bar."""
        pygame.draw.rect(
            self.screen, COLOR_DARK_GRAY, (0, 0, self.window_width, TOP_MENU_HEIGHT)
        )

        menu_items = ["New", "Load", "Export", "Resize", "Test", "Quit"]
        item_width = 80
        x = TOOLBAR_WIDTH

        for item in menu_items:
            # Draw button
            button_rect = pygame.Rect(x, 5, item_width - 10, TOP_MENU_HEIGHT - 10)
            pygame.draw.rect(self.screen, COLOR_GRAY, button_rect)
            pygame.draw.rect(self.screen, COLOR_WHITE, button_rect, 2)

            # Draw text
            text = self.small_font.render(item, True, COLOR_WHITE)
            text_rect = text.get_rect(center=button_rect.center)
            self.screen.blit(text, text_rect)

            x += item_width

    def draw_toolbar(self):
        """Draw the left toolbar with available tools."""
        pygame.draw.rect(
            self.screen,
            COLOR_DARK_GRAY,
            (0, TOP_MENU_HEIGHT, TOOLBAR_WIDTH, self.window_height),
        )

        y = TOP_MENU_HEIGHT + 5
        tool_height = 35

        for i, tool in enumerate(TOOLS):
            # Highlight selected tool
            if i == self.state.selected_tool:
                pygame.draw.rect(
                    self.screen, COLOR_BLUE, (5, y, TOOLBAR_WIDTH - 10, tool_height - 2)
                )
            else:
                pygame.draw.rect(
                    self.screen, COLOR_GRAY, (5, y, TOOLBAR_WIDTH - 10, tool_height - 2)
                )

            # Draw tool icon
            icon_rect = pygame.Rect(10, y + 5, 25, 25)
            pygame.draw.rect(self.screen, COLOR_WHITE, icon_rect)
            pygame.draw.rect(self.screen, COLOR_BLACK, icon_rect, 1)
            self._draw_tool_icon(tool, icon_rect)

            # Draw tool name and symbol
            text = self.small_font.render(
                f"{tool.symbol} - {tool.name}", True, COLOR_WHITE
            )
            self.screen.blit(text, (40, y + 8))

            # Draw keyboard shortcut if available
            if tool.keyboard_shortcut:
                shortcut_text = self.small_font.render(
                    f"[{tool.keyboard_shortcut}]", True, COLOR_LIGHT_GRAY
                )
                self.screen.blit(shortcut_text, (TOOLBAR_WIDTH - 40, y + 8))

            y += tool_height

    def _draw_tool_icon(self, tool: EditorTool, rect: pygame.Rect):
        """Draw a toolbar icon that matches the in-grid visuals."""
        if tool.is_erase:
            pygame.draw.rect(self.screen, COLOR_WHITE, rect)
            pygame.draw.line(self.screen, COLOR_RED, rect.topleft, rect.bottomright, 3)
            pygame.draw.line(self.screen, COLOR_RED, rect.topright, rect.bottomleft, 3)
            return

        if tool.is_agent:
            self._draw_agent(rect, 0)
            return

        if tool.conveyor_direction is not None:
            if "Player" in tool.name:
                self._draw_player_conveyor(rect, tool.conveyor_direction)
            else:
                self._draw_item_conveyor(rect, tool.conveyor_direction)
            return

        ingredient_idx = self._get_tool_ingredient_index(tool)
        if ingredient_idx is not None:
            self._draw_ingredient_pile(rect, ingredient_idx)
            return

        if tool.static_object is None:
            self._draw_generic(rect, 0)
            return

        if DEPENDENCIES_AVAILABLE:
            if tool.static_object == StaticObject.WALL:
                self._draw_wall(rect)
            elif tool.static_object == StaticObject.POT:
                self._draw_pot(rect)
            elif tool.static_object == StaticObject.GOAL:
                self._draw_goal(rect)
            elif tool.static_object == StaticObject.PLATE_PILE:
                self._draw_plate_pile(rect)
            elif tool.static_object == StaticObject.RECIPE_INDICATOR:
                self._draw_recipe_indicator(rect)
            else:
                self._draw_generic(rect, tool.static_object)
        else:
            if tool.static_object == FALLBACK_WALL:
                self._draw_wall(rect)
            elif tool.static_object == FALLBACK_POT:
                self._draw_pot(rect)
            elif tool.static_object == FALLBACK_GOAL:
                self._draw_goal(rect)
            elif tool.static_object == FALLBACK_PLATE_PILE:
                self._draw_plate_pile(rect)
            elif tool.static_object == FALLBACK_RECIPE:
                self._draw_recipe_indicator(rect)
            else:
                self._draw_generic(rect, tool.static_object)

    def _get_tool_ingredient_index(self, tool: EditorTool) -> Optional[int]:
        """Return ingredient index for a tool if it is an ingredient pile tool."""
        if "Ingredient" in tool.name:
            try:
                return int(tool.name.split()[-1])
            except ValueError:
                return None

        if tool.static_object is None:
            return None

        if DEPENDENCIES_AVAILABLE and StaticObject is not None:
            if StaticObject.is_ingredient_pile(tool.static_object):
                return tool.static_object - StaticObject.INGREDIENT_PILE_BASE
        elif FALLBACK_INGREDIENT_BASE <= tool.static_object < FALLBACK_ITEM_CONVEYOR:
            return tool.static_object - FALLBACK_INGREDIENT_BASE

        return None

    def draw_grid(self):
        """Draw the main grid editor."""
        grid_x_offset = TOOLBAR_WIDTH
        grid_y_offset = TOP_MENU_HEIGHT

        # Draw grid background
        grid_rect = pygame.Rect(
            grid_x_offset,
            grid_y_offset,
            self.state.width * self.tile_size,
            self.state.height * self.tile_size,
        )
        pygame.draw.rect(self.screen, COLOR_WHITE, grid_rect)

        # Draw grid cells
        for y in range(self.state.height):
            for x in range(self.state.width):
                cell_rect = pygame.Rect(
                    grid_x_offset + x * self.tile_size,
                    grid_y_offset + y * self.tile_size,
                    self.tile_size,
                    self.tile_size,
                )

                # Draw cell background
                pygame.draw.rect(self.screen, COLOR_WHITE, cell_rect)
                pygame.draw.rect(self.screen, COLOR_LIGHT_GRAY, cell_rect, 1)

                # Draw object in cell
                self.draw_cell_object(x, y, cell_rect)

        # Draw hover indicator
        if self.hover_pos:
            mx, my = self.hover_pos
            grid_x = mx - grid_x_offset
            grid_y = my - grid_y_offset

            if (
                0 <= grid_x < self.state.width * self.tile_size
                and 0 <= grid_y < self.state.height * self.tile_size
            ):
                cell_x = grid_x // self.tile_size
                cell_y = grid_y // self.tile_size
                hover_rect = pygame.Rect(
                    grid_x_offset + cell_x * self.tile_size,
                    grid_y_offset + cell_y * self.tile_size,
                    self.tile_size,
                    self.tile_size,
                )
                pygame.draw.rect(self.screen, COLOR_YELLOW, hover_rect, 3)

    def draw_cell_object(self, x: int, y: int, rect: pygame.Rect):
        """Draw the object at a specific grid cell, matching game visuals."""
        # Draw static object
        obj = self.state.static_objects[y, x]

        # Fill cell background
        pygame.draw.rect(self.screen, COLOR_WHITE, rect)

        if obj != 0:
            # Draw based on object type
            if DEPENDENCIES_AVAILABLE:
                if obj == StaticObject.WALL:
                    self._draw_wall(rect)
                elif obj == StaticObject.POT:
                    self._draw_pot(rect)
                elif obj == StaticObject.GOAL:
                    self._draw_goal(rect)
                elif obj == StaticObject.PLATE_PILE:
                    self._draw_plate_pile(rect)
                elif obj == StaticObject.RECIPE_INDICATOR:
                    self._draw_recipe_indicator(rect)
                elif StaticObject.is_ingredient_pile(obj):
                    ingredient_idx = obj - StaticObject.INGREDIENT_PILE_BASE
                    self._draw_ingredient_pile(rect, ingredient_idx)
                elif obj == StaticObject.ITEM_CONVEYOR:
                    self._draw_item_conveyor(rect, 0)
                elif obj == StaticObject.PLAYER_CONVEYOR:
                    self._draw_player_conveyor(rect, 0)
                else:
                    self._draw_generic(rect, obj)
            else:
                # Fallback: use numeric values
                if obj == FALLBACK_WALL:
                    self._draw_wall(rect)
                elif obj == FALLBACK_POT:
                    self._draw_pot(rect)
                elif obj == FALLBACK_PLATE_PILE:
                    self._draw_plate_pile(rect)
                elif obj == FALLBACK_GOAL:
                    self._draw_goal(rect)
                elif obj == FALLBACK_RECIPE:
                    self._draw_recipe_indicator(rect)
                elif FALLBACK_INGREDIENT_BASE <= obj < FALLBACK_ITEM_CONVEYOR:
                    self._draw_ingredient_pile(rect, obj - FALLBACK_INGREDIENT_BASE)
                elif obj == FALLBACK_ITEM_CONVEYOR:
                    self._draw_item_conveyor(rect, 0)
                elif obj == FALLBACK_PLAYER_CONVEYOR:
                    self._draw_player_conveyor(rect, 0)
                else:
                    self._draw_generic(rect, obj)

        # Draw conveyor if present
        if (y, x) in self.state.item_conveyors:
            direction = self.state.item_conveyors[(y, x)]
            self._draw_item_conveyor(rect, direction)

        if (y, x) in self.state.player_conveyors:
            direction = self.state.player_conveyors[(y, x)]
            self._draw_player_conveyor(rect, direction)

        # Draw agent if present
        if (x, y) in self.state.agent_positions:
            agent_idx = self.state.agent_positions.index((x, y))
            self._draw_agent(rect, agent_idx)

        # Draw grid lines
        pygame.draw.rect(self.screen, COLOR_LIGHT_GRAY, rect, 1)

    def _draw_wall(self, rect: pygame.Rect):
        """Draw wall sprite."""
        # Fill background
        pygame.draw.rect(self.screen, COLOR_GRAY, rect)
        # Draw grid pattern to show it's a wall
        for i in range(rect.x, rect.x + rect.width, 8):
            pygame.draw.line(
                self.screen, COLOR_LIGHT_GRAY, (i, rect.y), (i, rect.y + rect.height), 1
            )
        for j in range(rect.y, rect.y + rect.height, 8):
            pygame.draw.line(
                self.screen, COLOR_LIGHT_GRAY, (rect.x, j), (rect.x + rect.width, j), 1
            )

    def _draw_pot(self, rect: pygame.Rect):
        """Draw pot sprite matching the game visualization."""
        # Pot body - filled rectangle
        pot_rect = pygame.Rect(
            rect.x + rect.width * 0.1,
            rect.y + rect.height * 0.33,
            rect.width * 0.8,
            rect.height * 0.57,
        )
        pygame.draw.rect(self.screen, COLOR_GRAY, pot_rect)

        # Pot lid
        lid_rect = pygame.Rect(
            rect.x + rect.width * 0.1,
            rect.y + rect.height * 0.21,
            rect.width * 0.8,
            rect.height * 0.15,
        )
        pygame.draw.rect(self.screen, COLOR_DARK_GRAY, lid_rect)

        # Pot handle
        handle_rect = pygame.Rect(
            rect.x + rect.width * 0.4,
            rect.y + rect.height * 0.16,
            rect.width * 0.2,
            rect.height * 0.08,
        )
        pygame.draw.rect(self.screen, COLOR_DARK_GRAY, handle_rect)

    def _draw_plate_pile(self, rect: pygame.Rect):
        """Draw plate pile sprite matching the game visualization."""
        # Draw three stacked plates at different positions
        plate_positions = [
            (rect.centerx - rect.width * 0.15, rect.centery - rect.height * 0.15),
            (rect.centerx + rect.width * 0.15, rect.centery + rect.height * 0.05),
            (rect.centerx - rect.width * 0.05, rect.centery + rect.height * 0.2),
        ]

        for x, y in plate_positions:
            # Plate circle
            pygame.draw.circle(
                self.screen, COLOR_WHITE, (int(x), int(y)), int(rect.width * 0.2)
            )
            # Plate outline
            pygame.draw.circle(
                self.screen, COLOR_GRAY, (int(x), int(y)), int(rect.width * 0.2), 1
            )

    def _draw_goal(self, rect: pygame.Rect):
        """Draw delivery zone sprite matching the game visualization."""
        # Background
        pygame.draw.rect(self.screen, COLOR_GRAY, rect)
        # Inner green rectangle
        inner_rect = pygame.Rect(
            rect.x + rect.width * 0.1,
            rect.y + rect.height * 0.1,
            rect.width * 0.8,
            rect.height * 0.8,
        )
        pygame.draw.rect(self.screen, COLOR_GREEN, inner_rect)

    def _draw_recipe_indicator(self, rect: pygame.Rect):
        """Draw recipe indicator sprite matching the game visualization."""
        # Background
        pygame.draw.rect(self.screen, COLOR_GRAY, rect)
        # Inner brown rectangle
        inner_rect = pygame.Rect(
            rect.x + rect.width * 0.1,
            rect.y + rect.height * 0.1,
            rect.width * 0.8,
            rect.height * 0.8,
        )
        pygame.draw.rect(self.screen, COLOR_BROWN, inner_rect)

    def _draw_ingredient_pile(self, rect: pygame.Rect, ingredient_idx: int):
        """Draw ingredient pile sprite matching the game visualization."""
        # Background
        pygame.draw.rect(self.screen, COLOR_GRAY, rect)
        color = INGREDIENT_COLORS[ingredient_idx % len(INGREDIENT_COLORS)]

        # Draw ingredient circles in structured pile positions
        radius = int(rect.width * 0.15)
        positions = [
            (rect.centerx, rect.y + rect.height * 0.15),
            (rect.x + rect.width * 0.3, rect.y + rect.height * 0.4),
            (rect.x + rect.width * 0.8, rect.y + rect.height * 0.35),
            (rect.x + rect.width * 0.4, rect.y + rect.height * 0.8),
            (rect.x + rect.width * 0.75, rect.y + rect.height * 0.75),
        ]

        for x, y in positions:
            pygame.draw.circle(self.screen, color, (int(x), int(y)), radius)

    def _draw_item_conveyor(self, rect: pygame.Rect, direction: int = 0):
        """Draw item conveyor belt sprite matching the game visualization."""
        # Light grey background
        pygame.draw.rect(self.screen, COLOR_LIGHT_GRAY, rect)

        # Draw conveyor belt lines
        for i in range(4):
            y_offset = int(rect.y + rect.height * (0.15 + i * 0.2))
            line_height = int(rect.height * 0.1)
            pygame.draw.line(
                self.screen,
                COLOR_GRAY,
                (rect.x + int(rect.width * 0.05), y_offset),
                (rect.x + int(rect.width * 0.95), y_offset),
                2,
            )

        # Draw directional arrow
        center_x = rect.centerx
        center_y = rect.centery
        arrow_size = int(rect.width * 0.2)

        if direction == 2:  # Right (Direction.RIGHT)
            arrow_points = [
                (center_x + arrow_size, center_y),
                (center_x - arrow_size * 0.6, center_y - arrow_size * 0.6),
                (center_x - arrow_size * 0.6, center_y + arrow_size * 0.6),
            ]
        elif direction == 3:  # Left (Direction.LEFT)
            arrow_points = [
                (center_x - arrow_size, center_y),
                (center_x + arrow_size * 0.6, center_y - arrow_size * 0.6),
                (center_x + arrow_size * 0.6, center_y + arrow_size * 0.6),
            ]
        elif direction == 0:  # Up (Direction.UP)
            arrow_points = [
                (center_x, center_y - arrow_size),
                (center_x - arrow_size * 0.6, center_y + arrow_size * 0.6),
                (center_x + arrow_size * 0.6, center_y + arrow_size * 0.6),
            ]
        else:  # Down (Direction.DOWN = 1)
            arrow_points = [
                (center_x, center_y + arrow_size),
                (center_x - arrow_size * 0.6, center_y - arrow_size * 0.6),
                (center_x + arrow_size * 0.6, center_y - arrow_size * 0.6),
            ]

        pygame.draw.polygon(self.screen, COLOR_ORANGE, arrow_points)

    def _draw_player_conveyor(self, rect: pygame.Rect, direction: int = 0):
        """Draw player conveyor belt sprite matching the game visualization."""
        # Light blue background
        pygame.draw.rect(self.screen, (173, 216, 230), rect)  # light_blue color

        # Draw conveyor belt lines in blue
        for i in range(4):
            y_offset = int(rect.y + rect.height * (0.15 + i * 0.2))
            pygame.draw.line(
                self.screen,
                COLOR_BLUE,
                (rect.x + int(rect.width * 0.05), y_offset),
                (rect.x + int(rect.width * 0.95), y_offset),
                2,
            )

        # Draw directional arrow in cyan
        center_x = rect.centerx
        center_y = rect.centery
        arrow_size = int(rect.width * 0.2)

        if direction == 2:  # Right (Direction.RIGHT)
            arrow_points = [
                (center_x + arrow_size, center_y),
                (center_x - arrow_size * 0.6, center_y - arrow_size * 0.6),
                (center_x - arrow_size * 0.6, center_y + arrow_size * 0.6),
            ]
        elif direction == 3:  # Left (Direction.LEFT)
            arrow_points = [
                (center_x - arrow_size, center_y),
                (center_x + arrow_size * 0.6, center_y - arrow_size * 0.6),
                (center_x + arrow_size * 0.6, center_y + arrow_size * 0.6),
            ]
        elif direction == 0:  # Up (Direction.UP)
            arrow_points = [
                (center_x, center_y - arrow_size),
                (center_x - arrow_size * 0.6, center_y + arrow_size * 0.6),
                (center_x + arrow_size * 0.6, center_y + arrow_size * 0.6),
            ]
        else:  # Down (Direction.DOWN = 1)
            arrow_points = [
                (center_x, center_y + arrow_size),
                (center_x - arrow_size * 0.6, center_y - arrow_size * 0.6),
                (center_x + arrow_size * 0.6, center_y - arrow_size * 0.6),
            ]

        pygame.draw.polygon(self.screen, COLOR_CYAN, arrow_points)

    def _draw_agent(self, rect: pygame.Rect, agent_idx: int):
        """Draw agent sprite matching the game visualization."""
        color = AGENT_COLORS[agent_idx % len(AGENT_COLORS)]

        # Draw agent as triangle pointing right (direction 0)
        # Triangle coordinates based on game - pointing right
        center_x = rect.centerx
        center_y = rect.centery
        size = int(rect.width * 0.25)

        # Triangle pointing right
        points = [
            (center_x + size, center_y),  # Right point
            (center_x - size * 0.6, center_y - size * 0.7),  # Top-left point
            (center_x - size * 0.6, center_y + size * 0.7),  # Bottom-left point
        ]

        pygame.draw.polygon(self.screen, color, points)
        # Draw outline
        pygame.draw.polygon(self.screen, COLOR_WHITE, points, 2)

        # Draw agent number indicator
        number_text = self.small_font.render(str(agent_idx), True, COLOR_BLACK)
        text_rect = number_text.get_rect(center=(center_x + size * 0.3, center_y))
        self.screen.blit(number_text, text_rect)

    def _draw_generic(self, rect: pygame.Rect, obj: int):
        """Draw generic/unknown object."""
        pygame.draw.rect(self.screen, COLOR_GRAY, rect.inflate(-4, -4))
        text = self.small_font.render(f"#{obj}", True, COLOR_BLACK)
        text_rect = text.get_rect(center=rect.center)
        self.screen.blit(text, text_rect)

    def draw_info_panel(self):
        """Draw the right info panel."""
        available = self.window_width - TOOLBAR_WIDTH
        panel_x = TOOLBAR_WIDTH + available // 2
        panel_width = max(INFO_PANEL_WIDTH, self.window_width - panel_x)
        pygame.draw.rect(
            self.screen,
            COLOR_DARK_GRAY,
            (panel_x, TOP_MENU_HEIGHT, panel_width, self.window_height),
        )

        y = TOP_MENU_HEIGHT + 10

        def draw_wrapped_text(lines, x, start_y, max_chars=30):
            current_y = start_y
            for line in lines:
                if len(line) <= max_chars:
                    text = self.small_font.render(line, True, COLOR_WHITE)
                    self.screen.blit(text, (x, current_y))
                    current_y += 22
                    continue

                words = line.split()
                current_line = ""
                for word in words:
                    if len(current_line + word) <= max_chars:
                        current_line += word + " "
                    else:
                        text = self.small_font.render(
                            current_line.rstrip(), True, COLOR_WHITE
                        )
                        self.screen.blit(text, (x, current_y))
                        current_y += 22
                        current_line = word + " "
                if current_line:
                    text = self.small_font.render(
                        current_line.rstrip(), True, COLOR_WHITE
                    )
                    self.screen.blit(text, (x, current_y))
                    current_y += 22
            return current_y

        # Title
        title = self.title_font.render("Info", True, COLOR_WHITE)
        self.screen.blit(title, (panel_x + 10, y))
        y += 40

        # Layout info
        info_lines = [
            f"Size: {self.state.width}x{self.state.height}",
            f"Agents: {len(self.state.agent_positions)}",
            f"Name: {self.state.layout_name}",
            "",
        ]
        y = draw_wrapped_text(info_lines, panel_x + 10, y)

        # Selected tool details
        tool = TOOLS[self.state.selected_tool]
        selected_lines = [
            "Selected Tool:",
            f"{tool.symbol} - {tool.name}",
            tool.description,
            "",
        ]
        y = draw_wrapped_text(selected_lines, panel_x + 10, y)

        # Tool guide
        guide_lines = ["Tool Guide:"] + [f"{t.symbol} = {t.description}" for t in TOOLS]
        y = draw_wrapped_text(guide_lines, panel_x + 10, y)

        # Validation messages
        if self.validation_messages:
            y += 10
            validation_title = self.font.render("Validation:", True, COLOR_YELLOW)
            self.screen.blit(validation_title, (panel_x + 10, y))
            y += 25

            for msg in self.validation_messages[:10]:  # Limit to 10 messages
                # Green for success, red for errors
                msg_color = COLOR_GREEN if msg.startswith("✓") else COLOR_RED
                # Word wrap long messages
                if len(msg) > 30:
                    words = msg.split()
                    line = ""
                    for word in words:
                        if len(line + word) < 30:
                            line += word + " "
                        else:
                            text = self.small_font.render(line, True, msg_color)
                            self.screen.blit(text, (panel_x + 10, y))
                            y += 20
                            line = word + " "
                    if line:
                        text = self.small_font.render(line, True, msg_color)
                        self.screen.blit(text, (panel_x + 10, y))
                        y += 20
                else:
                    text = self.small_font.render(msg, True, msg_color)
                    self.screen.blit(text, (panel_x + 10, y))
                    y += 20

    def draw_export_dialog(self):
        """Draw the export dialog."""
        # Draw semi-transparent overlay
        overlay = pygame.Surface((self.window_width, self.window_height))
        overlay.set_alpha(200)
        overlay.fill(COLOR_BLACK)
        self.screen.blit(overlay, (0, 0))

        # Draw dialog box
        dialog_width = 600
        dialog_height = 400
        dialog_x = (self.window_width - dialog_width) // 2
        dialog_y = (self.window_height - dialog_height) // 2

        pygame.draw.rect(
            self.screen, COLOR_WHITE, (dialog_x, dialog_y, dialog_width, dialog_height)
        )
        pygame.draw.rect(
            self.screen,
            COLOR_BLUE,
            (dialog_x, dialog_y, dialog_width, dialog_height),
            3,
        )

        # Draw title
        title = self.title_font.render("Layout Exported!", True, COLOR_BLACK)
        self.screen.blit(title, (dialog_x + 20, dialog_y + 20))

        # Draw save path info
        y = dialog_y + 80
        saved_label = self.font.render("Saved to:", True, COLOR_BLACK)
        self.screen.blit(saved_label, (dialog_x + 20, y))
        y += 30
        path_text = self.small_font.render(self.export_text, True, COLOR_BLUE)
        self.screen.blit(path_text, (dialog_x + 20, y))

        # Draw close button
        close_button = pygame.Rect(
            dialog_x + dialog_width - 100, dialog_y + dialog_height - 50, 80, 35
        )
        pygame.draw.rect(self.screen, COLOR_BLUE, close_button)
        close_text = self.font.render("Close", True, COLOR_WHITE)
        close_text_rect = close_text.get_rect(center=close_button.center)
        self.screen.blit(close_text, close_text_rect)

        # Check for click on close button
        if pygame.mouse.get_pressed()[0]:
            mx, my = pygame.mouse.get_pos()
            if close_button.collidepoint(mx, my):
                self.show_export_dialog = False

    def draw_load_dialog(self):
        """Draw the load layout dialog."""
        if not DEPENDENCIES_AVAILABLE:
            return

        # Draw semi-transparent overlay
        overlay = pygame.Surface((self.window_width, self.window_height))
        overlay.set_alpha(200)
        overlay.fill(COLOR_BLACK)
        self.screen.blit(overlay, (0, 0))

        # Draw dialog box
        dialog_width = 500
        dialog_height = 500
        dialog_x = (self.window_width - dialog_width) // 2
        dialog_y = (self.window_height - dialog_height) // 2

        pygame.draw.rect(
            self.screen, COLOR_WHITE, (dialog_x, dialog_y, dialog_width, dialog_height)
        )
        pygame.draw.rect(
            self.screen,
            COLOR_BLUE,
            (dialog_x, dialog_y, dialog_width, dialog_height),
            3,
        )

        # Draw title
        title = self.title_font.render("Load Layout", True, COLOR_BLACK)
        self.screen.blit(title, (dialog_x + 20, dialog_y + 20))

        # List available layouts
        y = dialog_y + 70
        layout_names = list(overcooked_v3_layouts.keys())

        # Draw scrollable list of layouts
        for i, name in enumerate(layout_names[:15]):  # Show first 15
            button_rect = pygame.Rect(dialog_x + 20, y, dialog_width - 40, 25)

            # Highlight on hover
            mx, my = pygame.mouse.get_pos()
            if button_rect.collidepoint(mx, my):
                pygame.draw.rect(self.screen, COLOR_LIGHT_GRAY, button_rect)
                self.selected_layout_name = name

                # Handle click
                if pygame.mouse.get_pressed()[0]:
                    self.load_layout(name)
                    self.show_load_dialog = False
                    return
            else:
                pygame.draw.rect(self.screen, COLOR_WHITE, button_rect)

            pygame.draw.rect(self.screen, COLOR_GRAY, button_rect, 1)

            text = self.small_font.render(name, True, COLOR_BLACK)
            self.screen.blit(text, (dialog_x + 30, y + 5))

            y += 28

        # Draw close button
        close_button = pygame.Rect(
            dialog_x + dialog_width - 100, dialog_y + dialog_height - 50, 80, 35
        )
        pygame.draw.rect(self.screen, COLOR_RED, close_button)
        close_text = self.font.render("Cancel", True, COLOR_WHITE)
        close_text_rect = close_text.get_rect(center=close_button.center)
        self.screen.blit(close_text, close_text_rect)

        # Check for click on close button
        if pygame.mouse.get_pressed()[0]:
            mx, my = pygame.mouse.get_pos()
            if close_button.collidepoint(mx, my):
                self.show_load_dialog = False

    def draw_resize_dialog(self):
        """Draw the resize grid dialog."""
        # Draw semi-transparent overlay
        overlay = pygame.Surface((self.window_width, self.window_height))
        overlay.set_alpha(200)
        overlay.fill(COLOR_BLACK)
        self.screen.blit(overlay, (0, 0))

        # Draw dialog box
        dialog_width = 400
        dialog_height = 300
        dialog_x = (self.window_width - dialog_width) // 2
        dialog_y = (self.window_height - dialog_height) // 2

        pygame.draw.rect(
            self.screen, COLOR_WHITE, (dialog_x, dialog_y, dialog_width, dialog_height)
        )
        pygame.draw.rect(
            self.screen,
            COLOR_BLUE,
            (dialog_x, dialog_y, dialog_width, dialog_height),
            3,
        )

        # Draw title
        title = self.title_font.render("Resize Grid", True, COLOR_BLACK)
        self.screen.blit(title, (dialog_x + 20, dialog_y + 20))

        # Draw width input
        y = dialog_y + 80
        width_label = self.font.render("Width:", True, COLOR_BLACK)
        self.screen.blit(width_label, (dialog_x + 30, y))

        width_input_rect = pygame.Rect(dialog_x + 150, y, 150, 35)
        pygame.draw.rect(self.screen, COLOR_WHITE, width_input_rect)
        pygame.draw.rect(self.screen, COLOR_BLACK, width_input_rect, 2)

        width_text = self.font.render(self.resize_width_input, True, COLOR_BLACK)
        self.screen.blit(width_text, (width_input_rect.x + 10, width_input_rect.y + 5))

        # Draw height input
        y += 60
        height_label = self.font.render("Height:", True, COLOR_BLACK)
        self.screen.blit(height_label, (dialog_x + 30, y))

        height_input_rect = pygame.Rect(dialog_x + 150, y, 150, 35)
        pygame.draw.rect(self.screen, COLOR_WHITE, height_input_rect)
        pygame.draw.rect(self.screen, COLOR_BLACK, height_input_rect, 2)

        height_text = self.font.render(self.resize_height_input, True, COLOR_BLACK)
        self.screen.blit(
            height_text, (height_input_rect.x + 10, height_input_rect.y + 5)
        )

        # Draw error message if any
        if self.resize_error:
            error_text = self.small_font.render(self.resize_error, True, COLOR_RED)
            self.screen.blit(error_text, (dialog_x + 30, dialog_y + 200))

        # Draw buttons
        ok_button = pygame.Rect(dialog_x + 80, dialog_y + dialog_height - 50, 80, 35)
        cancel_button = pygame.Rect(
            dialog_x + 240, dialog_y + dialog_height - 50, 80, 35
        )

        # OK button
        pygame.draw.rect(self.screen, COLOR_GREEN, ok_button)
        ok_text = self.font.render("OK", True, COLOR_WHITE)
        ok_text_rect = ok_text.get_rect(center=ok_button.center)
        self.screen.blit(ok_text, ok_text_rect)

        # Cancel button
        pygame.draw.rect(self.screen, COLOR_RED, cancel_button)
        cancel_text = self.font.render("Cancel", True, COLOR_WHITE)
        cancel_text_rect = cancel_text.get_rect(center=cancel_button.center)
        self.screen.blit(cancel_text, cancel_text_rect)

        # Handle input
        mx, my = pygame.mouse.get_pos()

        if pygame.mouse.get_pressed()[0]:
            # Focus on width input
            if width_input_rect.collidepoint(mx, my):
                self.resize_input_mode = "width"
            # Focus on height input
            elif height_input_rect.collidepoint(mx, my):
                self.resize_input_mode = "height"
            # OK button
            elif ok_button.collidepoint(mx, my):
                try:
                    new_width = int(self.resize_width_input)
                    new_height = int(self.resize_height_input)

                    if not (MIN_GRID_WIDTH <= new_width <= MAX_GRID_WIDTH):
                        self.resize_error = (
                            f"Width must be {MIN_GRID_WIDTH}-{MAX_GRID_WIDTH}"
                        )
                        return

                    if not (MIN_GRID_HEIGHT <= new_height <= MAX_GRID_HEIGHT):
                        self.resize_error = (
                            f"Height must be {MIN_GRID_HEIGHT}-{MAX_GRID_HEIGHT}"
                        )
                        return

                    self.state.resize(new_width, new_height)
                    self.show_resize_dialog = False

                    # Recalc tile size for current window
                    self._recalc_tile_size()

                except ValueError:
                    self.resize_error = "Please enter valid integers"
            # Cancel button
            elif cancel_button.collidepoint(mx, my):
                self.show_resize_dialog = False

    def load_layout(self, layout_name: str):
        """Load an existing layout by name."""
        if not DEPENDENCIES_AVAILABLE:
            print("Cannot load: dependencies not available")
            return

        try:
            layout = overcooked_v3_layouts[layout_name]

            # Create new editor state from layout
            new_state = EditorState()
            new_state.width = layout.width
            new_state.height = layout.height
            new_state.static_objects = layout.static_objects.copy()
            new_state.agent_positions = layout.agent_positions.copy()
            new_state.recipes = (
                layout.possible_recipes.copy()
                if layout.possible_recipes
                else [[0, 0, 0]]
            )
            new_state.layout_name = layout_name + "_modified"

            # Convert conveyor info back to dicts
            for y, x, d in layout.item_conveyor_info:
                new_state.item_conveyors[(y, x)] = d

            for y, x, d in layout.player_conveyor_info:
                new_state.player_conveyors[(y, x)] = d

            self.state = new_state
            self.validation_messages = []

            # Recalc tile size for current window
            self._recalc_tile_size()

            print(f"✓ Loaded layout: {layout_name}")

        except Exception as e:
            print(f"Error loading layout {layout_name}: {e}")
            import traceback

            traceback.print_exc()
            self.validation_messages = [f"Load error: {e}"]


def main():
    """Entry point for the level editor."""
    # Load JaxMARL deps if available (needed for loading/validating real layouts)
    _load_jaxmarl_deps()

    print("=" * 60)
    print("OVERCOOKED V3 LEVEL EDITOR")
    print("=" * 60)
    print()

    if not DEPENDENCIES_AVAILABLE:
        print("WARNING: Running in limited mode without JaxMARL dependencies")
        print("Some features may not work correctly")
        print()

    print("Controls:")
    print("  Left Click: Place selected object")
    print("  Right Click: Erase object")
    print()
    print("Keyboard Shortcuts:")
    print("  W=Wall  P=Pot  B=Plate  X=Delivery  A=Agent  E=Erase")
    print("  0-9=Ingredients  Ctrl+Z=Undo  Ctrl+Y=Redo")
    print("  Ctrl+E=Export  Ctrl+T=Test")
    print()
    print("Menu: New, Load, Export, Test, Validate, Quit")
    print("=" * 60)
    print()

    editor = LevelEditor()
    editor.run()


if __name__ == "__main__":
    main()
