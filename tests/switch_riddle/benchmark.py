from jaxmarl import make
from rollout_menager import RolloutManager
from switch_riddle_nojax import SwitchRiddleNoJax
import time
import jax


def get_no_jax_time(parallel_envs, num_agents, max_steps):
    env = SwitchRiddleNoJax(num_agents, parallel_envs)
    _ = env.reset()
    actions = env.sample_actions()

    t0 = time.time()
    for i in range(max_steps):
        _ = env.step(actions)

    return time.time() - t0


def get_jax_time(parallel_envs, num_agents, max_steps):
    env = make("switch_riddle", num_agents=num_agents)
    wrapped_env = RolloutManager(env, batch_size=parallel_envs)
    key = jax.random.PRNGKey(0)

    # run once to make sure to compile
    key, key_r, key_a, key_s = jax.random.split(key, 4)
    _, state = wrapped_env.batch_reset(key_r)
    key_a = jax.random.split(key_a, env.num_agents)
    actions = {
        agent: wrapped_env.batch_sample(key_a[i], agent)
        for i, agent in enumerate(env.agents)
    }
    _, state, _, _, _ = wrapped_env.batch_step(key_s, state, actions)

    # run
    key, key_r = jax.random.split(key)
    _, state = wrapped_env.batch_reset(key_r)

    t0 = time.time()
    for i in range(max_steps):
        key, key_s = jax.random.split(key)
        (
            _,
            state,
            _,
            _,
            _,
        ) = wrapped_env.batch_step(key_s, state, actions)

    return time.time() - t0


def benchmark():
    parallel_envs = 100
    num_agents = 10
    max_steps = 1000

    jax_time = get_jax_time(parallel_envs, num_agents, max_steps)
    no_jax_time = get_no_jax_time(parallel_envs, num_agents, max_steps)

    print(
        "no jax time",
        no_jax_time,
        "jax time",
        jax_time,
        "speedup",
        no_jax_time / jax_time,
    )

    no_jax_per_step = (max_steps * parallel_envs) / no_jax_time
    jax_per_step = (max_steps * parallel_envs) / jax_time

    print(
        "no jax steps/sec",
        no_jax_per_step,
        "jax steps/sec",
        jax_per_step,
        "speedup",
        jax_per_step / no_jax_per_step,
    )


if __name__ == "__main__":
    benchmark()
