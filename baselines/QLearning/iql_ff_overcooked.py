import os
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import NamedTuple, Dict, Union, Any

import flax
import chex
import optax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from gymnax.wrappers.purerl import FlattenObservationWrapper, LogWrapper
import hydra
from omegaconf import OmegaConf
import gymnax
import flashbax as fbx
import wandb

from jaxmarl import make
from jaxmarl.wrappers.baselines import LogWrapper, CTRolloutManager
from jaxmarl.environments.overcooked import overcooked_layouts


class CNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=16, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))  # Flatten

        x = nn.Dense(features=64)(x)
        x = nn.relu(x)

        return x


class QNetwork(nn.Module):
    action_dim: int
    hidden_size: int = 512
    n_layers: int = 4

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = x.reshape(-1, *x.shape[-3:]) # three last dimensions are the conv, rest is batch
        x = CNN()(x)
        x = nn.Dense(self.action_dim)(x)
        return x


@chex.dataclass(frozen=True)
class Timestep:
    obs: dict
    actions: dict
    rewards: dict
    dones: dict


class CustomTrainState(TrainState):
    target_network_params: Any
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0


def make_train(config, env):

    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    eps_scheduler = optax.linear_schedule(
        config["EPS_START"],
        config["EPS_FINISH"],
        config["EPS_DECAY"] * config["NUM_UPDATES"],
    )

    def get_greedy_actions(q_vals, valid_actions):
        valid_actions = valid_actions.reshape(
            *[1] * len(q_vals.shape[:-1]), -1
        )  # reshape to match q_vals shape
        valid_q_vals = jnp.where(
            valid_actions.astype(bool), jax.lax.stop_gradient(q_vals), -1e6
        )
        return jnp.argmax(valid_q_vals, axis=-1)

    # epsilon-greedy exploration
    def eps_greedy_exploration(rng, q_vals, eps, valid_actions):

        rng_a, rng_e = jax.random.split(
            rng
        )  # a key for sampling random actions and one for picking

        greedy_actions = get_greedy_actions(q_vals, valid_actions)

        # pick random actions from the valid actions
        random_actions = jax.random.choice(
            rng_a,
            jnp.arange(valid_actions.shape[-1]),
            shape=greedy_actions.shape,
            p=valid_actions * 1.0 / jnp.sum(valid_actions, axis=-1),
        )

        chosed_actions = jnp.where(
            jax.random.uniform(rng_e, greedy_actions.shape)
            < eps,  # pick the actions that should be random
            random_actions,
            greedy_actions,
        )
        return chosed_actions

    def batchify(x: dict):
        return jnp.stack([x[agent] for agent in env.agents], axis=0)

    def unbatchify(x: jnp.ndarray):
        return {agent: x[i] for i, agent in enumerate(env.agents)}

    def train(rng):

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        wrapped_env = CTRolloutManager(env, batch_size=config["NUM_ENVS"], preprocess_obs=False)
        test_env = CTRolloutManager(
            env, batch_size=config["NUM_TEST_EPISODES"], preprocess_obs=False
        )  # batched env for testing (has different batch size)

        # INIT NETWORK AND OPTIMIZER
        network = QNetwork(
            action_dim=wrapped_env.max_action_space,
            hidden_size=config["HIDDEN_SIZE"],
            n_layers=config["N_LAYERS"],
        )

        def create_agent(rng):
            init_x = jnp.zeros(env.observation_space().shape)
            network_params = network.init(rng, init_x)

            lr_scheduler = optax.linear_schedule(
                config["LR"],
                1e-10,
                (config["NUM_MINI_EPOCHS"]) * config["NUM_UPDATES"],
            )

            lr = lr_scheduler if config.get("LR_LINEAR_DECAY", False) else config["LR"]

            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.radam(learning_rate=lr),
            )

            train_state = CustomTrainState.create(
                apply_fn=network.apply,
                params=network_params,
                target_network_params=network_params,
                tx=tx,
            )
            return train_state

        rng, _rng = jax.random.split(rng)
        train_state = create_agent(rng)

        # INIT BUFFER
        buffer = fbx.make_flat_buffer(
            max_length=int(config["BUFFER_SIZE"]),
            min_length=int(config["BUFFER_BATCH_SIZE"]),
            sample_batch_size=int(config["BUFFER_BATCH_SIZE"]),
            add_sequences=False,
            add_batch_size=int(config["NUM_ENVS"] * config["NUM_STEPS"]),
        )
        buffer = buffer.replace(
            init=jax.jit(buffer.init),
            add=jax.jit(buffer.add, donate_argnums=0),
            sample=jax.jit(buffer.sample),
            can_sample=jax.jit(buffer.can_sample),
        )

        _obs, _env_state = wrapped_env.batch_reset(_rng)
        _actions = {
            agent: wrapped_env.batch_sample(_rng, agent) for agent in env.agents
        }
        _obs, _, _rewards, _dones, _infos = wrapped_env.batch_step(
            _rng, _env_state, _actions
        )
        _timestep = Timestep(
            obs=_obs,
            actions=_actions,
            rewards=_rewards,
            dones=_dones,
        )
        _tiemstep_unbatched = jax.tree_map(
            lambda x: x[0], _timestep
        )  # remove the NUM_ENV dim
        buffer_state = buffer.init(_tiemstep_unbatched)

        # TRAINING LOOP
        def _update_step(runner_state, unused):

            train_state, buffer_state, expl_state, test_state, rng = runner_state

            # SAMPLE PHASE
            def _step_env(carry, _):
                last_obs, env_state, rng = carry
                rng, rng_a, rng_s = jax.random.split(rng, 3)
                q_vals = jax.vmap(network.apply, in_axes=(None, 0))(
                    train_state.params,
                    batchify(last_obs),  # (num_agents, num_envs, num_actions)
                )  # (num_agents, num_envs, num_actions)

                # explore
                eps = eps_scheduler(train_state.n_updates)
                actions = jax.vmap(eps_greedy_exploration, in_axes=(None, 0, None, 0))(
                    rng_a, q_vals, eps, batchify(wrapped_env.valid_actions_oh)
                )
                actions = unbatchify(actions)

                new_obs, new_env_state, rewards, dones, infos = wrapped_env.batch_step(
                    rng_s, env_state, actions
                )
                timestep = Timestep(
                    obs=last_obs,
                    actions=actions,
                    rewards=rewards,
                    dones=dones,
                )
                return (new_obs, new_env_state, rng), (timestep, infos)

            # step the env
            rng, _rng = jax.random.split(rng)
            carry, (timesteps, infos) = jax.lax.scan(
                _step_env,
                (*expl_state, _rng),
                None,
                config["NUM_STEPS"],
            )
            expl_state = carry[:2]

            train_state = train_state.replace(
                timesteps=train_state.timesteps
                + config["NUM_STEPS"] * config["NUM_ENVS"]
            )  # update timesteps count

            # BUFFER UPDATE
            timesteps = jax.tree_util.tree_map(
                lambda x: x.reshape(-1, *x.shape[2:]), timesteps
            )  # (num_envs*num_steps, ...)
            buffer_state = buffer.add(buffer_state, timesteps)

            # NETWORKS UPDATE
            def _learn_phase(carry, _):

                train_state, rng = carry
                rng, _rng = jax.random.split(rng)
                minibatch = buffer.sample(buffer_state, _rng).experience

                q_next_target = jax.vmap(network.apply, in_axes=(None, 0))(
                    train_state.target_network_params, batchify(minibatch.second.obs)
                )  # (num_agents, batch_size, ...)
                q_next_target = jnp.max(q_next_target, axis=-1)  # (batch_size,)
                
                if config["LOSS_TYPE"] == "vdn":
                    vdn_target = (
                        minibatch.first.rewards['__all__']
                        + (1 - minibatch.first.dones['__all__'])
                        * config["GAMMA"]
                        * jnp.sum(q_next_target, axis=0) # sum over agents
                    )
                else:
                    # iql loss
                    target = (
                        batchify(minibatch.first.rewards)
                        + (1 - batchify(minibatch.first.dones))
                        * config["GAMMA"]
                        * q_next_target
                    )

                def _loss_fn(params):
                    q_vals = jax.vmap(network.apply, in_axes=(None, 0))(
                        params, batchify(minibatch.first.obs)
                    )  # (num_agents, batch_size, ...)

                    # get logits of the chosen actions
                    chosen_action_q_vals = jnp.take_along_axis(
                        q_vals,
                        batchify(minibatch.first.actions)[..., jnp.newaxis],
                        axis=-1,
                    ).squeeze()  # (num_agents, batch_size, )

                    if config["LOSS_TYPE"] == "vdn":
                        chosen_action_q_vals = jnp.sum(chosen_action_q_vals, axis=0)
                        loss = jnp.mean((chosen_action_q_vals - vdn_target) ** 2)
                    else:
                        # iql: per agent loss
                        loss = jnp.mean((chosen_action_q_vals - target) ** 2)

                    return loss, chosen_action_q_vals.mean()

                (loss, qvals), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
                    train_state.params
                )
                train_state = train_state.apply_gradients(grads=grads)
                train_state = train_state.replace(
                    grad_steps=train_state.grad_steps + 1,
                )
                return (train_state, rng), (loss, qvals)

            rng, _rng = jax.random.split(rng)
            is_learn_time = (
                buffer.can_sample(buffer_state)
            ) & (  # enough experience in buffer
                train_state.timesteps > config["LEARNING_STARTS"]
            )
            (train_state, rng), (loss, qvals) = jax.lax.cond(
                is_learn_time,
                lambda train_state, rng: jax.lax.scan(
                    _learn_phase, (train_state, rng), None, config["NUM_MINI_EPOCHS"]
                ),
                lambda train_state, rng: (
                    (train_state, rng),
                    (
                        jnp.zeros(config["NUM_MINI_EPOCHS"]),
                        jnp.zeros(config["NUM_MINI_EPOCHS"]),
                    ),
                ),  # do nothing
                train_state,
                _rng,
            )

            # update target network
            train_state = jax.lax.cond(
                train_state.n_updates % config["TARGET_UPDATE_INTERVAL"] == 0,
                lambda train_state: train_state.replace(
                    target_network_params=optax.incremental_update(
                        train_state.params,
                        train_state.target_network_params,
                        config["TAU"],
                    )
                ),
                lambda train_state: train_state,
                operand=train_state,
            )

            # UPDATE METRICS
            train_state = train_state.replace(n_updates=train_state.n_updates + 1)
            metrics = {
                "returns": jnp.nanmean(
                    jnp.where(
                        infos["returned_episode"],
                        infos["returned_episode_returns"],
                        jnp.nan,
                    )
                ),
                "env_step": train_state.timesteps,
                "update_steps": train_state.n_updates,
                "grad_steps": train_state.grad_steps,
                "loss": loss.mean(),
                "qvals": qvals.mean(),
            }

            if config.get("TEST_DURING_TRAINING", True):
                rng, _rng = jax.random.split(rng)
                test_policy, test_obs, test_env_state = test_state
                q_vals = jax.vmap(network.apply, in_axes=(None, 0))(
                    test_policy.params, batchify(test_obs)
                )
                actions = jax.vmap(get_greedy_actions)(
                    q_vals,
                    batchify(test_env.valid_actions_oh),
                )
                actions = unbatchify(actions)
                test_new_obs, test_new_env_state, test_reward, test_done, test_info = (
                    test_env.batch_step(_rng, test_env_state, actions)
                )
                test_returns = jnp.nanmean(
                    jnp.where(
                        test_info["returned_episode"],
                        test_info["returned_episode_returns"],
                        jnp.nan,
                    )
                )
                test_policy = jax.lax.cond(
                    train_state.n_updates % config["TEST_POLICY_UPDATE_INTERVAL"] == 0,
                    lambda _: train_state,
                    lambda _: test_policy,
                    None,
                )
                test_state = (test_policy, test_new_obs, test_new_env_state)
                metrics["test_returns"] = test_returns

            # report on wandb if required
            if config.get("WANDB_LOG_DURING_TRAINING"):

                def callback(metrics):
                    wandb.log(metrics)

                jax.debug.callback(callback, metrics)

            runner_state = (train_state, buffer_state, expl_state, test_state, rng)

            return runner_state, None

        rng, _rng = jax.random.split(rng)
        obs, env_state = wrapped_env.batch_reset(_rng)
        expl_state = (obs, env_state)

        rng, _rng = jax.random.split(rng)
        test_obs, test_env_state = test_env.batch_reset(_rng)
        test_state = (train_state, test_obs, test_env_state)

        # train
        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, buffer_state, expl_state, test_state, _rng)

        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )

        return {"runner_state": runner_state, "metrics": metrics}

    return train


def single_run(config):

    print("Config:\n", OmegaConf.to_yaml(config))

    env_name = config["env"]["ENV_NAME"]
    alg_name = config["alg"]["ALG_NAME"]

    # overcooked needs a layout
    if "overcooked" in env_name.lower():
        config["env"]["ENV_KWARGS"]["layout"] = overcooked_layouts[
            config["env"]["ENV_KWARGS"]["layout"]
        ]
        env = make(config["env"]["ENV_NAME"], **config["env"]["ENV_KWARGS"])
        env = LogWrapper(env)
    else:
        env = make(config["env"]["ENV_NAME"], **config["env"]["ENV_KWARGS"])
        env = LogWrapper(env)

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=[
            alg_name.upper(),
            env_name.upper(),
            f"jax_{jax.__version__}",
        ],
        name=f"{alg_name}_{env_name}",
        config=config,
        mode=config["WANDB_MODE"],
    )

    rng = jax.random.PRNGKey(config["SEED"])

    if config["NUM_SEEDS"] > 1:
        rngs = jax.random.split(rng, config["NUM_SEEDS"])
        train_vjit = jax.jit(jax.vmap(make_train(config["alg"], env)))
        outs = jax.block_until_ready(train_vjit(rngs))
    else:
        outs = jax.jit(make_train(config["alg"]))(rng)


def tune(default_config):
    """Hyperparameter sweep with wandb."""
    import copy
    from multiprocessing import Process

    def wrapped_make_train():
        wandb.init(project=default_config["PROJECT"])

        # update the default params
        config = copy.deepcopy(default_config)
        for k, v in dict(wandb.config).items():
            config["alg"][k] = v

        print("running experiment with params:", config["alg"])

        rng = jax.random.PRNGKey(config["SEED"])
        # overcooked needs a layout
        if "overcooked" in default_config["env"]["ENV_NAME"].lower():
            config["env"]["ENV_KWARGS"]["layout"] = overcooked_layouts[
                config["env"]["ENV_KWARGS"]["layout"]
            ]
            env = make(config["env"]["ENV_NAME"], **config["env"]["ENV_KWARGS"])
            env = LogWrapper(env)
        else:
            env = make(config["env"]["ENV_NAME"], **config["env"]["ENV_KWARGS"])
            env = LogWrapper(env)

        if config["NUM_SEEDS"] > 1:
            rngs = jax.random.split(rng, config["NUM_SEEDS"])
            train_vjit = jax.jit(jax.vmap(make_train(config["alg"], env)))
            outs = jax.block_until_ready(train_vjit(rngs))
        else:
            outs = jax.jit(make_train(config["alg"]))(rng)

    sweep_config = {
        "name": "iql_overcooked",
        "method": "bayes",
        "metric": {
            "name": "test_returns",
            "goal": "maximize",
        },
        "parameters": {
            "LR": {
                "values": [
                    0.005,
                    0.001,
                    0.0005,
                    0.0001,
                    0.00005,
                ]
            },
            "LR_LINEAR_DECAY": {"values": [True, False]},
            "NUM_ENVS": {"values": [8, 32, 64]},
            "NUM_STEPS": {"values": [4, 8, 16, 32]},
            "TARGET_UPDATE_INTERVAL": {"values": [10, 100, 500, 1000]},
            "NUM_MINI_EPOCHS": {"values": [1, 2, 4, 8]},
            "BUFFER_BATCH_SIZE": {"values": [32, 64, 128, 256]},
            "EPS_FINISH": {"values": [0.05, 0.1]},
            "EPS_DECAY": {"values": [0.1, 0.2]},
        },
    }

    wandb.login()
    sweep_id = wandb.sweep(
        sweep_config, entity=default_config["ENTITY"], project=default_config["PROJECT"]
    )
    wandb.agent(sweep_id, wrapped_make_train, count=300)


@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(config):
    config = OmegaConf.to_container(config)
    print("Config:\n", OmegaConf.to_yaml(config))
    if config["HYP_TUNE"]:
        tune(config)
    else:
        single_run(config)


if __name__ == "__main__":
    main()
