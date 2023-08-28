"""
Short introduction to the package.

## Abstract base class
Uses the PettingZoo Parallel API. All agents act synchronously, with actions, 
observations, returns and dones passed as dictionaries keyed by agent names. 
The code can be found in `multiagentgymnax/multi_agent_env.py`

## Environment loop
Below is an example of a simple environment loop, using random actions.

"""

import jax 
from smax import make
from smax.viz.overcooked_visualizer import OvercookedVisualizer
from smax.environments.overcooked import Overcooked, overcooked_layouts, layout_grid_to_dict
import time

# Parameters + random keys
max_steps = 100
key = jax.random.PRNGKey(0)
key, key_r, key_a = jax.random.split(key, 3)

# Get one of the classic layouts (cramped_room, asymm_advantages, coord_ring, forced_coord, counter_circuit)
layout = overcooked_layouts["cramped_room"]

# # Or make your own!
# custom_layout_grid = """
# WWOWW
# WA  W
# B P X
# W   W
# WWOWW
# """
# layout = layout_grid_to_dict(custom_layout_grid)

# Instantiate environment
env = make('Overcooked', layout=layout, max_steps=max_steps)

obs, state = env.reset(key_r)
print('list of agents in environment', env.agents)

# Sample random actions
key_a = jax.random.split(key_a, env.num_agents)
actions = {agent: env.action_space(agent).sample(key_a[i]) for i, agent in enumerate(env.agents)}
print('example action dict', actions)

state_seq = []
for _ in range(max_steps):
    state_seq.append(state)
    # Iterate random keys and sample actions
    key, key_s, key_a = jax.random.split(key, 3)
    key_a = jax.random.split(key_a, env.num_agents)

    actions = {agent: env.action_space(agent).sample(key_a[i]) for i, agent in enumerate(env.agents)}

    # Step environment
    obs, state, rewards, dones, infos = env.step(key_s, state, actions)

viz = OvercookedVisualizer()
for s in state_seq:
    viz.render(env.agent_view_size, s, highlight=False)
    time.sleep(0.25)