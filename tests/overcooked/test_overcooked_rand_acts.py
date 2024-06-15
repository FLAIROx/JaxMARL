"""
Check that the environment can be reset and stepped with random actions.
TODO: replace this with proper unit tests.
"""
import jax
# import pytest 

from jaxmarl.environments.overcooked import Overcooked 

env = Overcooked()

def test_random_rollout():

    

    rng = jax.random.PRNGKey(0)
    rng, rng_reset = jax.random.split(rng)

    _, state = env.reset(rng_reset)
    
    for _ in range(10):
        rng, rng_act = jax.random.split(rng)
        rng_act = jax.random.split(rng_act, env.num_agents)
        actions = {a: env.action_space(a).sample(rng_act[i]) for i, a in enumerate(env.agents)}
        _, state, _, _, _ = env.step(rng, state, actions)
        

    
    