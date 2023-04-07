from pettingzoo.mpe import simple_world_comm_v2


# Simple world comm for 1000 steps, with randomly sampled actions

env = simple_world_comm_v2.parallel_env(max_cycles=1000)
obs = env.reset()

step=0
while env.agents:
    step += 1
    print('step', step)
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}  # this is where you would insert your policy
    observations, rewards, terminations, truncations, infos = env.step(actions)