import os
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import Any

from flax import struct
import chex
import optax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import hydra
from omegaconf import OmegaConf
import flashbax as fbx
import wandb

from flax.training import orbax_utils
from orbax.checkpoint import (
    PyTreeCheckpointer,
    CheckpointManagerOptions,
    CheckpointManager,
)

from jaxmarl import make
from jaxmarl.environments.smax import map_name_to_scenario
from jaxmarl.wrappers.baselines import SMAXLogWrapper, LogWrapper, CTRolloutManager
from jaxmarl.environments.overcooked import overcooked_layouts


class QNetwork(nn.Module):
    action_dim: int
    hidden_size: int = 512
    n_layers: int = 4
    norm_type: str = "batch_norm"

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False):
        if self.norm_type == "batch_norm":
            x = nn.BatchNorm(use_running_average=True)(x)
        else:
            x_dummy = nn.BatchNorm(use_running_average=True)(x)
        for l in range(1, self.n_layers - 1):
            x = nn.Dense(self.hidden_size)(x)
            if self.norm_type == "batch_norm":
                x = nn.BatchNorm(use_running_average=True)(x)
            elif self.norm_type == "layer_norm":
                x = nn.LayerNorm()(x)
            x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return x


@chex.dataclass(frozen=True)
class Transition:
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    next_obs: chex.Array
    next_avail_actions: chex.Array


class CustomTrainState(TrainState):
    batch_stats: Any
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

    def get_greedy_actions(q_vals, valid_actions):
        unavail_actions = 1 - valid_actions
        q_vals = q_vals - (unavail_actions * 1e10)
        return jnp.argmax(q_vals, axis=-1)

    # epsilon-greedy exploration
    def eps_greedy_exploration(rng, q_vals, eps, valid_actions):

        rng_a, rng_e = jax.random.split(
            rng
        )  # a key for sampling random actions and one for picking

        greedy_actions = get_greedy_actions(q_vals, valid_actions)

        # pick random actions from the valid actions
        def get_random_actions(rng, val_action):
            return jax.random.choice(
                rng,
                jnp.arange(val_action.shape[-1]),
                p=val_action * 1.0 / jnp.sum(val_action, axis=-1),
            )

        _rngs = jax.random.split(rng_a, valid_actions.shape[0])
        random_actions = jax.vmap(get_random_actions)(_rngs, valid_actions)

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
        wrapped_env = CTRolloutManager(
            env, batch_size=config["NUM_ENVS"], preprocess_obs=config["PREPROCESS_OBS"]
        )
        test_env = CTRolloutManager(
            env,
            batch_size=config["NUM_TEST_EPISODES"],
            preprocess_obs=config["PREPROCESS_OBS"],
        )  # batched env for testing (has different batch size)

        # INIT NETWORK AND OPTIMIZER
        network = QNetwork(
            action_dim=wrapped_env.max_action_space,
            hidden_size=config["HIDDEN_SIZE"],
            n_layers=config["N_LAYERS"],
            norm_type=config["NORM_TYPE"],
        )

        def create_agent(rng):
            init_x = jnp.zeros((1, wrapped_env.obs_size))
            network_variables = network.init(rng, init_x, train=False)

            lr_scheduler = optax.linear_schedule(
                config["LR"],
                1e-10,
                config["NUM_EPOCHS"]
                * config["NUM_MINIBATCHES"]
                * config["NUM_UPDATES"],
            )

            lr = lr_scheduler if config.get("LR_LINEAR_DECAY", False) else config["LR"]

            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.radam(learning_rate=lr),
            )

            train_state = CustomTrainState.create(
                apply_fn=network.apply,
                params=network_variables["params"],
                batch_stats=network_variables["batch_stats"],
                tx=tx,
            )
            return train_state

        rng, _rng = jax.random.split(rng)
        train_state = create_agent(rng)

        # TRAINING LOOP
        def _update_step(runner_state, unused):

            train_state, expl_state, test_state, rng = runner_state

            # SAMPLE PHASE
            def _step_env(carry, _):
                last_obs, env_state, rng = carry
                rng, rng_a, rng_s = jax.random.split(rng, 3)
                q_vals = jax.vmap(network.apply, in_axes=(None, 0, None))(
                    {
                        "params": train_state.params,
                        "batch_stats": train_state.batch_stats,
                    },
                    batchify(last_obs),  # (num_agents, num_envs, num_actions)
                    False,
                )  # (num_agents, num_envs, num_actions)

                # explore
                avail_actions = wrapped_env.get_valid_actions(env_state.env_state)

                eps = eps_scheduler(train_state.n_updates)
                _rngs = jax.random.split(rng_a, env.num_agents)
                new_action = jax.vmap(eps_greedy_exploration, in_axes=(0, 0, None, 0))(
                    _rngs, q_vals, eps, batchify(avail_actions)
                )
                new_action = unbatchify(new_action)

                new_obs, new_env_state, reward, new_done, info = wrapped_env.batch_step(
                    rng_s, env_state, new_action
                )

                # get the next available action
                next_avail_actions = wrapped_env.get_valid_actions(new_env_state.env_state)

                transition = Transition(
                    obs=batchify(last_obs),  # (num_agents, num_envs, obs_shape)
                    action=batchify(new_action),  # (num_agents, num_envs,)
                    reward=reward["__all__"][np.newaxis],  # (num_envs,)
                    done=new_done["__all__"][np.newaxis],  # (num_envs,)
                    next_obs=batchify(new_obs),  # (num_agents, num_envs, obs_shape)
                    next_avail_actions=batchify(next_avail_actions),  # (num_agents, num_envs, num_actions)
                )
                return (new_obs, new_env_state, rng), (transition, info)

            # step the env
            rng, _rng = jax.random.split(rng)
            (*expl_state, rng), (transitions, infos) = jax.lax.scan(
                _step_env,
                (*expl_state, _rng),
                None,
                config["NUM_STEPS"],
            )
            expl_state = tuple(expl_state)

            train_state = train_state.replace(
                timesteps=train_state.timesteps
                + config["NUM_STEPS"] * config["NUM_ENVS"]
            )  # update timesteps count

            transitions = jax.tree_map(
                lambda x: jnp.swapaxes(x, 1, 2), transitions
            )  # (num_steps, num_envs, num_agents, ...)

            # NETWORKS UPDATE
            def _learn_epoch(carry, _):
                train_state, rng = carry

                def _learn_phase(carry, minibatch):

                    train_state, rng = carry
                    minibatch = jax.tree_map(
                        lambda x: jnp.swapaxes(x, 0, 1), minibatch
                    )  # (num agents, batch size)

                    def _loss_fn(params):
                        agent_in = jnp.concatenate(
                            (minibatch.obs, minibatch.next_obs), axis=1
                        )  # (num_agents, batch_size*2, obs_shape)
                        agent_in = agent_in.reshape(
                            -1, agent_in.shape[-1]
                        )  # (num_agents*batch_size*2, obs_shape)
                        all_q_vals, updates = network.apply(
                            {"params": params, "batch_stats": train_state.batch_stats},
                            agent_in,
                            train=False,
                            mutable=["batch_stats"],
                        )  # (num_agents, batch_size*2, num_actions)
                        all_q_vals = all_q_vals.reshape(
                            env.num_agents, -1, wrapped_env.max_action_space
                        )  # (num_agents, batch_size*2, num_actions)

                        q_vals, q_next = jnp.split(
                            all_q_vals, 2, axis=1
                        )  # (num_agents, batch_size, num_actions), (num_agents, batch_size, num_actions)
                        
                        q_next = jax.lax.stop_gradient(q_next)
                        unavailable_actions = 1 - minibatch.next_avail_actions
                        q_next = q_next - (unavailable_actions * 1e10)
                        
                        q_next = jnp.max(q_next, axis=-1)  # (num_agents, batch_size,)
                        vdn_q_next = jnp.sum(q_next, axis=0)  # (batch_size,)
                        target = (
                            minibatch.reward[0]
                            + (1 - minibatch.done[0]) * config["GAMMA"] * vdn_q_next
                        )

                        chosen_action_qvals = jnp.take_along_axis(
                            q_vals,
                            jnp.expand_dims(minibatch.action, axis=-1),
                            axis=-1,
                        ).squeeze(
                            axis=-1
                        )  # (num_agents, batch_size,)

                        vdn_chosen_action_qvals = jnp.sum(
                            chosen_action_qvals, axis=0
                        )  # (batch_size,)

                        loss = jnp.mean(
                            (vdn_chosen_action_qvals - jax.lax.stop_gradient(target))
                            ** 2
                        )
                        return loss, (updates, chosen_action_qvals)

                    (loss, (updates, qvals)), grads = jax.value_and_grad(
                        _loss_fn, has_aux=True
                    )(train_state.params)
                    train_state = train_state.apply_gradients(grads=grads)
                    train_state = train_state.replace(
                        grad_steps=train_state.grad_steps + 1,
                        batch_stats=updates["batch_stats"],
                    )
                    return (train_state, rng), (loss, qvals)

                def preprocess_transition(x, rng):
                    x = x.reshape(
                        -1, *x.shape[2:]
                    )  # num_steps*num_envs (batch_size), ...
                    x = jax.random.permutation(
                        rng, x, axis=1
                    )  # shuffle the transitions
                    x = x.reshape(
                        config["NUM_MINIBATCHES"], -1, *x.shape[1:]
                    )  # num_mini_updates, batch_size/num_mini_updates, ...
                    return x

                rng, _rng = jax.random.split(rng)
                minibatches = jax.tree_util.tree_map(
                    lambda x: preprocess_transition(x, _rng),
                    transitions,
                )  # num_minibatches, batch_size, num_agents, ...

                rng, _rng = jax.random.split(rng)
                (train_state, rng), (loss, qvals) = jax.lax.scan(
                    _learn_phase, (train_state, rng), minibatches
                )

                return (train_state, rng), (loss, qvals)

            rng, _rng = jax.random.split(rng)
            (train_state, rng), (loss, qvals) = jax.lax.scan(
                _learn_epoch, (train_state, rng), None, config["NUM_EPOCHS"]
            )

            train_state = train_state.replace(n_updates=train_state.n_updates + 1)
            metrics = {
                "env_step": train_state.timesteps,
                "updates": train_state.n_updates,
                "grad_steps": train_state.grad_steps,
                "loss": loss.mean(),
                "qvals": qvals.mean(),
                "returns": infos["returned_episode_returns"].mean(),
            }

            if config.get("TEST_DURING_TRAINING", True):
                rng, _rng = jax.random.split(rng)
                test_state = jax.lax.cond(
                    train_state.n_updates % config["TEST_INTERVAL"] == 0,
                    lambda _: get_greedy_metrics(_rng, train_state),
                    lambda _: test_state,
                    operand=None,
                )
                metrics.update({"test_" + k: v for k, v in test_state.items()})

            # report on wandb if required
            if config.get("WANDB_LOG_DURING_TRAINING"):

                def callback(metrics):
                    metrics["env_step"] = metrics["updates"] * config["NUM_STEPS"] * config[
                        "NUM_ENVS"
                    ] # metrics["env_step"] can overflow in int32
                    wandb.log(metrics)

                jax.debug.callback(callback, metrics)

            runner_state = (train_state, tuple(expl_state), test_state, rng)

            return runner_state, None

        def get_greedy_metrics(rng, train_state):
            if not config.get("TEST_DURING_TRAINING", True):
                return None
            """Help function to test greedy policy during training"""

            def _greedy_env_step(step_state, unused):
                env_state, last_obs, rng = step_state
                rng, key_s = jax.random.split(rng)
                _obs = batchify(last_obs)
                q_vals = jax.vmap(network.apply, in_axes=(None, 0))(
                    {
                        "params": train_state.params,
                        "batch_stats": train_state.batch_stats,
                    },
                    _obs,
                )
                valid_actions = test_env.get_valid_actions(env_state.env_state)
                actions = get_greedy_actions(q_vals, batchify(valid_actions))
                actions = unbatchify(actions)
                obs, env_state, rewards, dones, infos = test_env.batch_step(
                    key_s, env_state, actions
                )
                step_state = (env_state, obs, rng)
                return step_state, (rewards, dones, infos)

            rng, _rng = jax.random.split(rng)
            init_obs, env_state = test_env.batch_reset(_rng)
            rng, _rng = jax.random.split(rng)
            step_state = (
                env_state,
                init_obs,
                _rng,
            )
            step_state, (rewards, dones, infos) = jax.lax.scan(
                _greedy_env_step, step_state, None, config["NUM_TEST_STEPS"]
            )
            metrics = jax.tree_map(
                lambda x: jnp.nanmean(
                    jnp.where(
                        infos["returned_episode"],
                        x,
                        jnp.nan,
                    )
                ),
                infos,
            )
            return metrics

        rng, _rng = jax.random.split(rng)
        test_state = get_greedy_metrics(_rng, train_state)

        rng, _rng = jax.random.split(rng)
        obs, env_state = wrapped_env.batch_reset(_rng)
        expl_state = (obs, env_state)

        # train
        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, expl_state, test_state, _rng)

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
    # smac init neeeds a scenario
    if "smax" in env_name.lower():
        config["env"]["ENV_KWARGS"]["scenario"] = map_name_to_scenario(
            config["env"]["MAP_NAME"]
        )
        env_name = f"{config['env']['ENV_NAME']}_{config['env']['MAP_NAME']}"
        env = make(config["env"]["ENV_NAME"], **config["env"]["ENV_KWARGS"])
        env = SMAXLogWrapper(env)
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

    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_vjit = jax.jit(jax.vmap(make_train(config["alg"], env)))
    outs = jax.block_until_ready(train_vjit(rngs))


def tune(default_config):
    """Hyperparameter sweep with wandb."""
    import copy
    from multiprocessing import Process

    def wrapped_make_train():
        wandb.init(project=default_config["PROJECT"])

        def run_experiment():
            # update the default params
            config = copy.deepcopy(default_config)
            for k, v in dict(wandb.config).items():
                config["alg"][k] = v

            print("running experiment with params:", config["alg"])

            rng = jax.random.PRNGKey(config["SEED"])

            if config["NUM_SEEDS"] > 1:
                rngs = jax.random.split(rng, config["NUM_SEEDS"])
                train_vjit = jax.jit(jax.vmap(make_train(config["alg"])))
                outs = jax.block_until_ready(train_vjit(rngs))
            else:
                outs = jax.jit(make_train(config["alg"]))(rng)

        p = Process(target=run_experiment)
        p.start()
        p.join(default_config["EXP_TIME_LIMIT"])  # Timeo

        if p.is_alive():
            print("Experiment timed out.")
            p.terminate()
            p.join()

    sweep_config = {
        "name": "pqn_hanabi",
        "method": "bayes",
        "metric": {
            "name": "returns",
            "goal": "maximize",
        },
        "parameters": {
            "LR": {
                "values": [
                    0.001,
                    0.0005,
                    0.0001,
                    0.00005,
                    0.00001,
                ]
            },
            #"NUM_ENVS": {"values": [512, 1024, 2048]},
            #"NUM_STEPS": {"values": [1, 16, 64]},
            #"NUM_MINIBATCHES": {"values": [1, 2, 4]},
            #"NUM_EPOCHS": {"values": [1, 2, 4]},
            "LR_DECAY": {"values": [True, False]},
            "NORM_TYPE": {"values": ["none", "batch_norm", "layer_norm"]},
            "HIDDEN_SIZE": {"values": [512, 1024]},
            "N_LAYERS": {"values": [3, 4, 5, 6]},
        },
    }

    wandb.login()
    sweep_id = wandb.sweep(
        sweep_config, entity=default_config["ENTITY"], project=default_config["PROJECT"]
    )
    wandb.agent(sweep_id, wrapped_make_train, count=100)


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
