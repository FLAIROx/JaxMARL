import jax
import jax.numpy as jnp
from multiagentgymnax.mpe.mpe_simple_world_comm import SimpleWorldCommEnv
from pettingzoo.mpe import simple_world_comm_v2
import time

# Simple world comm for 1000 steps, with randomly sampled actions
max_steps = 1000

env = simple_world_comm_v2.parallel_env(max_cycles=max_steps)
obs = env.reset()

#step=0
start_time = time.time()
actions = {agent: env.action_space(agent).sample() for agent in env.agents} 
#while env.agents:
    #step += 1
for _ in range(max_steps):    
    #actions = {agent: env.action_space(agent).sample() for agent in env.agents}  # this is where you would insert your policy
    observations, rewards, terminations, truncations, infos = env.step(actions)
zoo_time = time.time() - start_time


key = jax.random.PRNGKey(0)

env = SimpleWorldCommEnv()

key, key_r = jax.random.split(key)
state = env.reset_env(key_r)

#obs = env.observation(0, state)
#print('obs', obs.shape, obs)

mock_action = jnp.array([[0.0, 0.0, 1.0, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0]])

actions = jnp.repeat(mock_action[None], repeats=env.num_agents, axis=0).squeeze()


start_time = time.time()
for _ in range(max_steps):
    key, key_a, key_s = jax.random.split(key, 3)
    #actions = test_policy(key_a, state)
    #print('actions', actions)
    obs, state, rew, _ = env.step_env(key_s, state, actions)
    #env.render(state)
    #print('rew', rew)
jax_time = time.time() - start_time

print('zoo time', zoo_time, 'jax time', jax_time, 'speedup', zoo_time/jax_time)



