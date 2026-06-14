# Overcooked V3 Level Editor

A visual, interactive level editor for creating Overcooked V3 layouts with click-to-place functionality, live preview, and immediate test-play capabilities.

## Features

- 🎨 **Visual Grid Editor**: Click to place objects, right-click to erase
- 🎮 **Live Preview**: See your layout rendered in real-time
- 🏃 **Test Play**: Launch an interactive game with your layout instantly
- ✅ **Validation**: Automatic checking for missing required elements
- 📤 **Export to Code**: Generate ready-to-paste layout code
- 📥 **Load Existing**: Edit any registered Overcooked V3 layout
- ↩️ **Undo/Redo**: Full undo/redo support (Ctrl+Z / Ctrl+Y)
- 📚 **Built-in Legend**: Always-visible symbol reference

## Installation

Ensure you have JaxMARL dependencies installed:

```bash
cd JaxMARL
pip install -e .[algs]
```

Required dependencies:
- `pygame` (for the editor interface)
- `jax`, `jaxlib` (for layout validation and test play)
- `numpy` (for grid operations)

## Quick Start

### Launch the Editor

From the JaxMARL root directory:

```bash
python -m jaxmarl.tools.layout_editor_v3
```

### Creating Your First Layout

1. **Select a tool** from the left toolbar (or use keyboard shortcuts)
2. **Click on the grid** to place objects
3. **Right-click** to erase
4. **Click "Test"** in the menu to try your level
5. **Click "Export"** when ready to save

## Controls

### Mouse Controls

- **Left Click**: Place selected object
- **Right Click**: Erase object at position
- **Menu Clicks**: Access New, Load, Export, Test, Validate, Quit

### Keyboard Shortcuts

| Key | Tool | Description |
|-----|------|-------------|
| `W` | Wall | Counter/wall tiles |
| `P` | Pot | Cooking pot |
| `B` | Plate | Plate (bowl) pile |
| `X` | Delivery | Delivery zone (goal) |
| `A` | Agent | Agent starting position |
| `R` | Recipe | Recipe indicator (randomized) |
| `0`-`9` | Ingredients | Ingredient piles (0=onion, 1=tomato, etc.) |
| `E` | Erase | Eraser tool |

### Editor Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+Z` | Undo |
| `Ctrl+Y` | Redo |
| `Ctrl+N` | New layout |
| `Ctrl+E` | Export layout |
| `Ctrl+T` | Test play |

## Symbol Legend

### Basic Objects

- `W` = Wall/Counter
- `P` = Pot (cooking)
- `B` = Plate (Bowl) Pile
- `X` = Delivery Zone (goal)
- `A` = Agent Start Position
- `R` = Recipe Indicator
- `0`-`9` = Ingredient Piles (0=onion, 1=tomato, 2=lettuce, etc.)

### Conveyors

**Item Conveyors** (move items):
- `>` = moves right
- `<` = moves left
- `^` = moves up
- `v` = moves down

**Player Conveyors** (push agents):
- `]` = pushes right
- `[` = pushes left
- `{` = pushes up
- `}` = pushes down

## Workflow

### 1. Create Layout

Use the visual editor to design your level:

```
Example 5x4 layout:
WWPWW
0A AX
W   W
WBWWW
```

### 2. Validate

Click **Validate** to check for:
- ✓ At least one agent
- ✓ At least one delivery zone
- ✓ At least one ingredient source
- ✓ Plate pile (warning if missing)
- ✓ Pot (warning if missing)
- ✓ Within MAX_POTS, MAX_CONVEYORS limits

### 3. Test Play

Click **Test** to launch an interactive game:
- **Agent 0**: WASD + Space (interact)
- **Agent 1**: Arrow keys + Enter (interact)
- **R**: Reset game
- **Q/ESC**: Quit and return to editor

### 4. Export

Click **Export** to generate code:

```python
# Generated code snippet
my_custom_layout = """
WWPWW
0A AX
W   W
WBWWW
"""

overcooked_v3_layouts["my_custom_layout"] = Layout.from_string(
    my_custom_layout,
    possible_recipes=[[0, 0, 0]],
    swap_agents=False
)
```

### 5. Add to Project

Copy the exported code to:
```
jaxmarl/environments/overcooked_v3/layouts.py
```

Add your layout to the `overcooked_v3_layouts` dictionary (around line 280).

### 6. Use in Training

```python
from jaxmarl import make

env = make('overcooked_v3', layout='my_custom_layout')
```

Or update training configs:
```yaml
# baselines/IPPO/config/overcooked.yaml
ENV_KWARGS:
  layout: "my_custom_layout"
```

## Advanced Features

### Loading Existing Layouts

1. Click **Load** in the menu
2. Select a layout from the list
3. Edit and save as new layout (automatically appends "_modified")

### Recipes

Recipes define valid soup combinations (3 ingredients each):

```python
[[0, 0, 0]]           # 3 onions
[[1, 1, 1]]           # 3 tomatoes
[[0, 1, 1]]           # 1 onion + 2 tomatoes
[[0, 0, 0], [1, 1, 1]]  # Either 3 onions OR 3 tomatoes
```

Currently recipes are set in the code (default: `[[0, 0, 0]]`). Future versions will include a recipe editor dialog.

### Grid Resizing

Currently layouts use default 7x5 grid. Future versions will include:
- Resize buttons in the UI
- Custom dimensions input
- Auto-resize on load

### Conveyor Belts

**Item Conveyors** (cyan): Move held items in the specified direction

**Player Conveyors** (purple): Push agents in the specified direction

Both conveyor types have maximum limits defined in `settings.py`:
- `MAX_ITEM_CONVEYORS = 16`
- `MAX_PLAYER_CONVEYORS = 8`

## Troubleshooting

### "Cannot import JaxMARL dependencies"

Ensure you're in the correct environment and have installed dependencies:
```bash
pip install -e .[algs]
```

### "Layout has validation errors"

Check the info panel on the right for specific issues:
- Missing agents, delivery zones, or ingredients
- Too many pots or conveyors
- Invalid recipe format

### Test Play doesn't launch

Ensure:
- Layout passes validation (click Validate first)
- JAX is properly installed
- No other pygame windows are open

### Export code doesn't work

Make sure to:
1. Copy the entire exported code block
2. Add to `layouts.py` (not a new file)
3. Place before the `overcooked_v3_layouts` dictionary
4. Also add the dictionary entry shown in the export

## File Structure

```
JaxMARL/
├── jaxmarl/
│   ├── tools/
│   │   ├── __init__.py
│   │   └── layout_editor_v3.py               # Main editor
│   └── environments/
│       └── overcooked_v3/
│           ├── layouts.py                     # Add layouts here
│           └── ...
└── tests/
    └── overcooked_v3/
        └── test_layout_utils.py              # Layout utility method tests
```

## Tips & Best Practices

### Layout Design Tips

1. **Start with borders**: Place walls around the perimeter
2. **Agent reachability**: Ensure all objects are reachable by agents
3. **Pot-to-delivery distance**: Balance challenge and reward
4. **Ingredient variety**: Use multiple ingredient types for complex recipes
5. **Test early**: Use Test Play frequently during design

### Common Patterns

**Simple Kitchen** (good for testing):
```
WWPWW
0A AX
W   W
WBWWW
```

**Forced Cooperation** (agents must work together):
```
WWWPW
0 WAP
0AW W
B W W
WWWXW
```

**Conveyor Challenge** (adds movement dynamics):
```
WWWPWWW
0A >  X
W  v  W
W  > AW
WWWBWWW
```

### Performance Considerations

- Keep layouts under 20x20 for rendering performance
- Limit conveyors to avoid complex interactions
- Test with both agents in different starting positions

## Contributing

When creating levels for the team repository, include:

1. **Descriptive name**: e.g., `forced_coord_v2`, `conveyor_circuit`
2. **Clear recipes**: Document what recipes are possible
3. **Validation**: Ensure layout passes all validation checks
4. **Testing**: Run at least a few test episodes
5. **Documentation**: Add comments explaining level design goals

## Future Enhancements

Planned features:
- [ ] Recipe editor dialog
- [ ] Grid resize controls in UI
- [ ] Save/load editor projects (JSON)
- [ ] Multiple undo branches
- [ ] Copy/paste regions
- [ ] Symmetry tools
- [ ] Layout templates
- [ ] Batch export multiple layouts

## Support

For issues or questions:
1. Check validation messages in the editor
2. Review this README
3. Check `jaxmarl/environments/overcooked_v3/README.md`
4. Examine existing layouts in `layouts.py` for examples

## License

Part of the JaxMARL project. See repository LICENSE for details.

---

**Happy level designing! 🎮👨‍🍳**
