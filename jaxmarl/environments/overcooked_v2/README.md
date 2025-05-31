# OvercookedV2

OvercookedV2 is an extended version of the original Overcooked environment. It introduces meaningful partial observability and increased stochasticity to enable more complex coordination challenges.

## Key Features

- Configurable agent view radius
- Multiple ingredients and recipes
- Asymmetric information through recipe indicators
- Randomized starting positions and directions
- Grounded communication via button recipe indicators
- Flexible layout creation

## Custom Layouts

Create custom layouts using ASCII strings:

```python
layout = """
WWPWW
0A A1
L   R
WBWXW
"""
recipes = [[0,0,1], [0,1,1]]
custom_layout = Layout.from_string(layout, possible_recipes=recipes)
```

## Pre-configured Layouts

See `layouts.py` for a variety of pre-configured layouts:

- Adaptations of original Overcooked layouts
- Extended Cat-Dog problem layouts
- Test-time coordination challenge layouts

## Observations

Observations are structured as a width x height x num_channels tensor, with partial observability based on the configured view radius.

## Rewards

- 20 points for correct deliveries
- Optional -20 points for incorrect deliveries
- Shaped rewards for actions aligned with the current recipe

## Visualization and Interactive Play

- JIT-compiled rendering pipeline for efficient episode visualization
- Interactive mode for playing alongside trained policies

