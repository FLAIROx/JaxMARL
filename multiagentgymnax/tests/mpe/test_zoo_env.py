import jax
import jax.numpy as jnp
import numpy as np
import pettingzoo
from pettingzoo.mpe import simple_world_comm_v2
#from multiagentgymnax.u

from multiagentgymnax.mpe.mpe_simple_world_comm import SimpleWorldCommEnv

num_episodes, num_steps, tolerance = 10, 25, 1e-4


"""
state = State(
    p_pos=p_pos,
    p_vel=jnp.zeros((self.num_entities, self.dim_p)),
    c=jnp.zeros((self.num_agents, self.dim_c)),
    done=jnp.full((self.num_agents), False),
    step=0
)                        
"""

def np_state_to_jax(env_zoo, env_jax):

    from multiagentgymnax.mpe.mpe_base_env import State

    p_pos = np.zeros((env_jax.num_entities, env_jax.dim_p))
    p_vel = np.zeros((env_jax.num_entities, env_jax.dim_p))
    c = np.zeros((env_jax.num_entities, env_jax.dim_c))
    print('--', env_zoo.aec_env.agents) # gives list of agent names
    print('--', env_zoo.aec_env.env.world.agents)
    for agent in env_zoo.aec_env.env.world.agents:
        a_idx = env_jax.a_to_i[agent.name]
        p_pos[a_idx] = agent.state.p_pos
        p_vel[a_idx] = agent.state.p_vel
        c[a_idx] = agent.state.c


    for landmark in env_zoo.aec_env.env.world.landmarks:
        l_idx = env_jax.l_to_i[landmark.name]
        print('name', landmark.name)
        p_pos[l_idx] = landmark.state.p_pos


        
    print('p_pos', p_pos)

    state = {
        "p_pos": p_pos,
        "p_vel": p_vel,
        "c": c,
        "step": env_zoo.aec_env.env.steps,
        "done": np.full((env_jax.num_agents), False),
    }
    
    return State(**state)




def test_step(zoo_env_name):
    key = jax.random.PRNGKey(0)
    env_zoo = simple_world_comm_v2.parallel_env(max_cycles=25, continuous_actions=True)
    zoo_obs = env_zoo.reset()
    env_jax = SimpleWorldCommEnv()
    env_params = env_jax.default_params

    for ep in range(num_episodes):
        obs = env_zoo.reset()

        for s in range(num_steps):
            actions = {agent: env_zoo.action_space(agent).sample() for agent in env_zoo.agents}
            state = np_state_to_jax(env_zoo, env_jax)
            obs_zoo, rew_zoo, done_zoo, _, _ = env_zoo.step(actions)
            key, key_step = jax.random.split(key)
            obs_jax, state_jax, rew_jax, done_jax, _ = env_jax.step(key_step, state, actions)
            raise

if __name__=="__main__":

    test_step("simple_world_comm_v2")