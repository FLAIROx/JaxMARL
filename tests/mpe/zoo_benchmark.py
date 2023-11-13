"""
Simple benchmarking of the speed of JaxMARL vs PettingZoo's MPE environments.
"""

import jax
import jax.numpy as jnp
from jaxmarl.environments.mpe.simple_world_comm import SimpleWorldCommMPE
#from ._test_utils.rollout_manager import RolloutManager
from pettingzoo.mpe import simple_v3, simple_tag_v3, simple_world_comm_v3
import time

# Simple world comm for 1000 steps, with randomly sampled actions
max_steps = 1000
num_envs = 70


### PETTING ZOO BENCHMARK
env = simple_world_comm_v3.parallel_env(max_cycles=max_steps)
obs = env.reset()

start_time = time.time()
actions = {agent: env.action_space(agent).sample() for agent in env.agents} 

print('obs spaces', env.observation_spaces, env.action_spaces)
#while env.agents:
    #step += 1
for _ in range(max_steps):    
    #actions = {agent: env.action_space(agent).sample() for agent in env.agents}  # this is where you would insert your policy
    print('actions', actions)
    observations, rewards, terminations, truncations, infos = env.step(actions)
    print('terms', terminations)
    raise
zoo_time = time.time() - start_time


raise
### JAX BENCHMARK
key = jax.random.PRNGKey(0)

env = SimpleWorldCommMPE()

rollout_manager = RolloutManager(env)


key, key_r = jax.random.split(key)
key_r = jax.random.split(key_r, num_envs)
#state = env.reset_env(key_r)
obs, state = rollout_manager.batch_reset(key_r)

#obs = env.observation(0, state)
#print('obs', obs.shape, obs)

mock_action = jnp.array([[0.0, 0.0, 1.0, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0]])

actions = jnp.repeat(mock_action[None], repeats=env.num_agents, axis=0).squeeze()
actions = jnp.repeat(actions[None], repeats=num_envs, axis=0)

start_time = time.time()
for _ in range(max_steps):
    key, key_a, key_s = jax.random.split(key, 3)
    #actions = test_policy(key_a, state)
    #print('actions', actions)
    key_s = jax.random.split(key_s, num_envs)
    obs, state, rew, dones, _ = rollout_manager.batch_step(key_s, state, actions)
    #env.step_env(key_s, state, actions)
    #env.render(state)
    #print('rew', rew)
jax_time = time.time() - start_time

print('zoo time', zoo_time, 'jax time', jax_time, 'speedup', zoo_time/jax_time)

zoo_per_step = max_steps/zoo_time
jax_per_step = (max_steps*num_envs)/jax_time

print('zoo steps/sec', zoo_per_step , 'jax steps/sec', jax_per_step, 'speedup', jax_per_step/zoo_per_step)

