"""
Short introduction to the package.

## Abstract base class
Uses the PettingZoo Parallel API. All agents act synchronously, with actions, 
observations, returns and dones passed as dictionaries keyed by agent names. 
The code can be found in `multiagentgymnax/multi_agent_env.py`

The class follows an identical structure to that of `gymnax` with one execption. 
The class is instatiated with `num_agents`, defining the number of agents within the environment.

## Environment loop
Below is an example of a simple environment loop, using random actions.

"""

import jax 
from smax import make
from smax.viz.visualizer import Visualizer, MiniSMACVisualizer

# Parameters + random keys
max_steps = 100
key = jax.random.PRNGKey(3)
key, key_r, key_a = jax.random.split(key, 3)

# Instantiate environment
with jax.disable_jit(False):
    env, params = make('HeuristicEnemyMiniSMAC')
    obs, state = env.reset(key_r)
    print('list of agents in environment', env.agents)

    # Sample random actions
    key_a = jax.random.split(key_a, env.num_agents)
    actions = {agent: env.action_space(agent).sample(key_a[i]) for i, agent in enumerate(env.agents)}
    print('example action dict', actions)


    state_seq = []
    for _ in range(max_steps):
        # Iterate random keys and sample actions
        key, key_s, key_seq = jax.random.split(key, 3)
        key_a = jax.random.split(key_seq, env.num_agents)
        actions = {agent: env.action_space(agent).sample(key_a[i]) for i, agent in enumerate(env.agents)}

        state_seq.append((key_seq, state, actions))
        # state_seq.append(state)
        # Step environment
        obs, state, rewards, dones, infos = env.step(key_s, state, actions)


viz = MiniSMACVisualizer(env, state_seq, env.default_params)
viz.animate(view=False, save_fname="output.gif")