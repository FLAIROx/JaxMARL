import jax 
from jax import numpy as jnp
import chex
import numpy as np
from jaxmarl.environments.hanabi.hanabi import HanabiEnv

def main():
    rng = jax.random.PRNGKey(0)
    env = HanabiEnv()
    obs, env_state = env.reset(rng)
    print(env.action_encoding.values())

    env.render(env_state)
    obs_s = env.get_obs_str(env_state, include_belief=True, best_belief=5)
    print(obs_s)

    actions = jnp.array([15, 7, 13, 19])

    for a in actions:
        actions = {agent:a for agent in env.agents}
        obs, new_env_state, rewards, dones, infos  = env.step_env(rng, env_state, actions)
        obs_s = env.get_obs_str(new_env_state, env_state, a, include_belief=True, best_belief=5)
        print(obs_s)
        env_state = new_env_state


if __name__=='__main__':
    main()
