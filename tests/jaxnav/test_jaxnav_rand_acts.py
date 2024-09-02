"""
Check that the environment can be reset and stepped with random actions.
TODO: replace this with proper unit tests.
"""
import jax
import pytest 

from jaxmarl.environments.jaxnav import JaxNav 


@pytest.mark.parametrize(
    ("num_agents",),
    [
        (1,),
        (4,),
        (9,),
    ],
)
def test_random_rollout(num_agents: int):
    env = JaxNav(num_agents=num_agents)
    rng = jax.random.PRNGKey(0)
    rng, rng_reset = jax.random.split(rng)

    _, state = env.reset(rng_reset)
    
    for _ in range(10):
        rng, rng_act = jax.random.split(rng)
        rng_act = jax.random.split(rng_act, env.num_agents)
        actions = {a: env.action_space(a).sample(rng_act[i]) for i, a in enumerate(env.agents)}
        _, state, _, _, _ = env.step(rng, state, actions)
        
test_random_rollout(1)
test_random_rollout(4)
    
    