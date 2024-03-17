from typing import Sequence, NamedTuple, Any
from flax.linen.initializers import constant, orthogonal
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
import numpy as np
import optax
import distrax
from jaxmarl import make
from jaxmarl.environments.smax import map_name_to_scenario, Scenario
import time


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: dict, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}

EXTRA_SCENARIOS = {
    "5m": Scenario(jnp.zeros((10,), dtype=jnp.uint8), num_allies=5, num_enemies=5, smacv2_position_generation=False, smacv2_unit_type_generation=False),
    "50m": Scenario(jnp.zeros((100,), dtype=jnp.uint8), num_allies=50, num_enemies=50, smacv2_position_generation=False, smacv2_unit_type_generation=False),
    "500m": Scenario(jnp.zeros((1000,), dtype=jnp.uint8), num_allies=500, num_enemies=500, smacv2_position_generation=False, smacv2_unit_type_generation=False),
    "5000m": Scenario(jnp.zeros((10000,), dtype=jnp.uint8), num_allies=5000, num_enemies=5000, smacv2_position_generation=False, smacv2_unit_type_generation=False)
}


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


def make_benchmark(config):
    if config["MAP_NAME"] in EXTRA_SCENARIOS:
        scenario = EXTRA_SCENARIOS[config["MAP_NAME"]]
    else:
        scenario = map_name_to_scenario(config["MAP_NAME"])
    env = make(config["ENV_NAME"], scenario=scenario, **config["ENV_KWARGS"])
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    network = ActorCritic(
        env.action_space(env.agents[0]).n, activation=config["ACTIVATION"]
    )

    def benchmark(rng):
        def init_runner_state(rng):
            # INIT NETWORK
            network = ActorCritic(
                env.action_space(env.agents[0]).n, activation=config["ACTIVATION"]
            )
            rng, _rng = jax.random.split(rng)
            init_x = jnp.zeros(env.observation_space(env.agents[0]).shape)
            params = network.init(_rng, init_x)
            # INIT ENV
            rng, _rng = jax.random.split(rng)
            reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
            obsv, env_state = jax.vmap(env.reset)(reset_rng)

            return (params, env_state, obsv, rng)

        def env_step(runner_state, unused):
            params, env_state, last_obs, rng = runner_state

            # SELECT ACTION
            rng, _rng = jax.random.split(rng)

            obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])

            if config["ACTION_SELECTION"] == "nn":
                pi, _ = network.apply(params, obs_batch)
                # transform using the available actions
                avail_actions = jax.vmap(env.get_avail_actions)(env_state)
                avail_actions = jax.lax.stop_gradient(
                    batchify(avail_actions, env.agents, config["NUM_ACTORS"])
                )
                probs = jnp.where(avail_actions, pi.probs, 1e-10)
                probs = probs / probs.sum(axis=-1)[:, None]
                pi = distrax.Categorical(probs=probs)

                action = pi.sample(seed=_rng)
            elif config["ACTION_SELECTION"] == "random":
                avail_actions = jax.vmap(env.get_avail_actions)(env_state)
                avail_actions = jax.lax.stop_gradient(
                    batchify(avail_actions, env.agents, config["NUM_ACTORS"])
                )
                avail_actions = avail_actions.reshape(
                    config["NUM_ENVS"], env.num_agents, -1
                )
                logits = jnp.log(
                    jnp.ones((avail_actions.shape[-1],))
                    / jnp.sum(jnp.ones((avail_actions.shape[-1],)))
                )
                action = jax.random.categorical(
                    _rng, logits=logits, shape=avail_actions.shape[:-1]
                )
                action = action.reshape(config["NUM_ACTORS"], -1)
            else:
                raise ValueError("ACTION_SELECTION must be 'random' or 'nn'")

            env_act = unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents)
            env_act = {k: v.squeeze() for k, v in env_act.items()}

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, config["NUM_ENVS"])
            obsv, env_state, _, _, info = jax.vmap(env.step)(
                rng_step, env_state, env_act
            )
            info = jax.tree_map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
            runner_state = (params, env_state, obsv, rng)
            return runner_state, None

        rng, init_rng = jax.random.split(rng)
        runner_state = init_runner_state(init_rng)
        runner_state = jax.lax.scan(env_step, runner_state, None, config["NUM_STEPS"])
        return runner_state

    return benchmark


def main():
    config = {
        "NUM_STEPS": 128,
        "NUM_ENVS": 4,
        "ACTIVATION": "relu",
        "MAP_NAME": "50m",
        "ENV_KWARGS": {
            "map_width": 32,
            "map_height": 32,
        },
        "ENV_NAME": "HeuristicEnemySMAX",
        "NUM_SEEDS": 1,
        "SEED": 0,
        "ACTION_SELECTION": "random"
    }
    benchmark_fn = make_benchmark(config)
    rng = jax.random.PRNGKey(config["SEED"])
    benchmark_jit = jax.jit(benchmark_fn).lower(rng).compile()
    before = time.perf_counter_ns()
    runner_state = jax.block_until_ready(benchmark_jit(rng))
    after = time.perf_counter_ns()
    num_steps = config["NUM_ENVS"] * config["NUM_STEPS"]
    total_time = (after - before) / 1e9
    print(f"Total Time (s): {total_time}")
    print(f"Total Steps: {num_steps}")
    print(f"SPS: {num_steps / total_time}")


if __name__ == "__main__":
    main()
