

import jax
import jax.numpy as jnp
from multiagentgymnax.mpe.mpe_simple_world_comm import SimpleWorldCommEnv
from multiagentgymnax.mpe.train_utils import RolloutManager
from pettingzoo.mpe import simple_world_comm_v2
import time

# Simple world comm for 1000 steps, with randomly sampled actions
max_steps = 1000
num_envs = 70


### PETTING ZOO BENCHMARK
env = simple_world_comm_v2.parallel_env(max_cycles=max_steps)
obs = env.reset()

start_time = time.time()
actions = {agent: env.action_space(agent).sample() for agent in env.agents} 
#while env.agents:
    #step += 1
for _ in range(max_steps):    
    #actions = {agent: env.action_space(agent).sample() for agent in env.agents}  # this is where you would insert your policy
    observations, rewards, terminations, truncations, infos = env.step(actions)
    print('obs', observations)
    raise
zoo_time = time.time() - start_time

""" 
petting zoo uses dicts with explicit keys [adversay, agent]
We should use the same format.

Can then vmap over agent classes when needed with seperate funcs.

Padding is fine within the env but the external API should not differ.

"""