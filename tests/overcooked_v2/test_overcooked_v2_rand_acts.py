import jax
from jaxmarl import make

env_parmas = {
    "layout" : "grounded_coord_simple",
    "agent_view_size": 2,
    "negative_rewards": True,
    "sample_recipe_on_delivery": True,
    "random_agent_positions": True
}

env = make("overcooked_v2", **env_parmas)

def test_random_rollout_overcooked_v2():
    rng = jax.random.PRNGKey(0)
    rng, rng_reset = jax.random.split(rng)

    _, state = env.reset(rng_reset)
    
    for _ in range(10):
        rng, rng_act = jax.random.split(rng)
        rng_act = jax.random.split(rng_act, env.num_agents)
        actions = {a: env.action_space(a).sample(rng_act[i]) for i, a in enumerate(env.agents)}
        _, state, _, _, _ = env.step(rng, state, actions)
