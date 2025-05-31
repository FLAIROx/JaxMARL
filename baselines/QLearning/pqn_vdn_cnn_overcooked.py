"""
Specific to this implementation: CNN network and Reward Shaping Annealing as per Overcooked paper.
"""
import os
import copy
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import Any

import chex
import optax
import flax.linen as nn
from flax.training.train_state import TrainState
import hydra
from omegaconf import OmegaConf
import wandb

from jaxmarl import make
from jaxmarl.environments.smax import map_name_to_scenario
from jaxmarl.wrappers.baselines import (
    SMAXLogWrapper,
    MPELogWrapper,
    LogWrapper,
    CTRolloutManager,
)
from jaxmarl.environments.overcooked import overcooked_layouts


class CNN(nn.Module):
    norm_type: str = "layer_norm"

    @nn.compact
    def __call__(self, x, train=False):

        activation = nn.relu

        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        elif self.norm_type == "batch_norm":
            normalize = lambda x: nn.BatchNorm(use_running_average=not train)(x)
        else:
            normalize = lambda x: x

        x = nn.Conv(
            features=32,
            kernel_size=(5, 5),
        )(x)
        x = normalize(x)
        x = activation(x)
        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
        )(x)
        x = normalize(x)
        x = activation(x)
        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
        )(x)
        x = normalize(x)
        x = activation(x)
        x = x.reshape((x.shape[0], -1))  # Flatten

        return x


class QNetwork(nn.Module):
    action_dim: int
    hidden_size: int = 64
    num_layers: int = 2
    norm_input: bool = False
    norm_type: str = "layer_norm"

    @nn.compact
    def __call__(self, x: jnp.ndarray, train=False):

        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        elif self.norm_type == "batch_norm":
            normalize = lambda x: nn.BatchNorm(use_running_average=not train)(x)
        else:
            normalize = lambda x: x

        if self.norm_input:
            x = nn.BatchNorm(use_running_average=not train)(x)
        else:
            x_dummy = nn.BatchNorm(use_running_average=not train)(x)

        x = CNN(norm_type=self.norm_type)(x, train=train)
        for l in range(self.num_layers):
            x = nn.Dense(self.hidden_size)(x)
            x = normalize(x)
            x = nn.relu(x)

        q_vals = nn.Dense(self.action_dim)(x)
        
        return q_vals


@chex.dataclass(frozen=True)
class Transition:
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    avail_actions: chex.Array
    q_vals: chex.Array

class CustomTrainState(TrainState):
    batch_stats: Any
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0


def make_train(config, env):

    assert (
        (config["NUM_ENVS"]*config["NUM_STEPS"]) % config["NUM_MINIBATCHES"] == 0
    ), "NUM_ENVS*NUM_STEPS must be divisible by NUM_MINIBATCHES"

    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    eps_scheduler = optax.linear_schedule(
        config["EPS_START"],
        config["EPS_FINISH"],
        config["EPS_DECAY"] * config["NUM_UPDATES"],
    )

    rew_shaping_anneal = optax.linear_schedule(
        init_value=1.0, end_value=0.0, transition_steps=config["REW_SHAPING_HORIZON"]
    )

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

        original_seed = rng[0]

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        wrapped_env = CTRolloutManager(
            env, batch_size=config["NUM_ENVS"], preprocess_obs=False
        )
        test_env = CTRolloutManager(
            env,
            batch_size=config["TEST_NUM_ENVS"],
            preprocess_obs=False,
        )

        # INIT NETWORK AND OPTIMIZER
        network = QNetwork(
            action_dim=wrapped_env.max_action_space,
            hidden_size=config["HIDDEN_SIZE"],
            num_layers=config["NUM_LAYERS"],
            norm_type=config["NORM_TYPE"],
            norm_input=config["NORM_INPUT"],
        )

        def create_agent(rng):
            init_x = jnp.zeros((1, *env.observation_space().shape))
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

                # add shaped reward
                shaped_reward = info.pop("shaped_reward")
                shaped_reward["__all__"] = batchify(shaped_reward).sum(axis=0)
                reward = jax.tree.map(
                    lambda x, y: x + y * rew_shaping_anneal(train_state.timesteps),
                    reward,
                    shaped_reward,
                )

                # get the next available action
                next_avail_actions = wrapped_env.get_valid_actions(
                    new_env_state.env_state
                )

                transition = Transition(
                    obs=batchify(last_obs),  # (num_agents, num_envs, obs_shape)
                    action=batchify(new_action),  # (num_agents, num_envs,)
                    reward=config.get("REW_SCALE", 1)
                    * reward["__all__"][np.newaxis],  # (num_envs,)
                    done=new_done["__all__"][np.newaxis],  # (num_envs,)
                    avail_actions=batchify(
                        next_avail_actions
                    ),  # (num_agents, num_envs, num_actions)
                    q_vals=q_vals,  # (num_agents, num_envs, num_actions)
                )
                return (new_obs, new_env_state, rng), (transition, info)

            # step the env
            # transitions: (num_steps, num_agents, num_envs, ...)
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

            last_obs, env_state = expl_state
            last_q = jax.vmap(network.apply, in_axes=(None, 0, None))(
                {
                    "params": train_state.params,
                    "batch_stats": train_state.batch_stats,
                },
                batchify(last_obs),  # (num_agents, num_envs, num_actions)
                False,
            )  # (num_agents, num_envs, num_actions)
            unavail_actions = 1 - batchify(
                wrapped_env.get_valid_actions(env_state.env_state)
            )
            last_q = last_q - (unavail_actions * 1e10)
            last_q = jnp.max(last_q, axis=-1)  # (num_agents, num_envs)
            last_q = last_q.sum(axis=0)  # (num_envs) vdn sum

            def _compute_targets(last_q, q_vals, reward, done):
                def _get_target(lambda_returns_and_next_q, rew_q_done):
                    reward, q, done = rew_q_done # (num_envs) except for q (num_agents, num_envs, num_actions)
                    lambda_returns, next_q = lambda_returns_and_next_q # (num_envs)
                    target_bootstrap = reward + config["GAMMA"] * (1 - done) * next_q
                    delta = lambda_returns - next_q
                    lambda_returns = (
                        target_bootstrap + config["GAMMA"] * config["LAMBDA"] * delta
                    )
                    lambda_returns = (1 - done) * lambda_returns + done * reward
                    next_q = jnp.max(q, axis=-1)
                    next_q = next_q.sum(axis=0) # sum over agents (vdn)
                    return (lambda_returns, next_q), lambda_returns

                lambda_returns = reward[-1] + config["GAMMA"] * (1 - done[-1]) * last_q
                last_q = jnp.max(q_vals[-1], axis=-1)
                last_q = jnp.sum(last_q, axis=0) # sum over agents
                _, targets = jax.lax.scan(
                    _get_target,
                    (lambda_returns, last_q),
                    jax.tree.map(lambda x: x[:-1], (reward, q_vals, done)),
                    reverse=True,
                )
                targets = jnp.concatenate((targets, lambda_returns[np.newaxis]))
                return targets

            if config["NUM_STEPS"] > 1: # q-lambda returns
                q_vals = transitions.q_vals - (1 - transitions.avail_actions) * 1e10 # mask invalid actions
                lambda_targets = _compute_targets(
                    last_q,
                    q_vals,  # vdn sum,
                    transitions.reward[:, 0],  # _all_
                    transitions.done[:, 0],  # _all_
                ).reshape(
                    -1
                )  # (num_steps*num_envs)
            else:  # standard 1 step qlearning
                lambda_targets = (
                    transitions.reward[-1, 0]
                    + (1 - transitions.done[-1, 0]) * config["GAMMA"] * last_q
                )

            # NETWORKS UPDATE
            def _learn_epoch(carry, _):
                train_state, rng = carry

                def _learn_phase(carry, minibatch_and_target):

                    # minibatch shape: num_agents, batch_size, ...
                    # target shape: batch_size
                    # with batch_size = num_envs/num_minibatches

                    train_state, rng = carry
                    minibatch, target = minibatch_and_target

                    def _loss_fn(params):
                        agent_in = minibatch.obs.reshape(
                            -1, *minibatch.obs.shape[2:]
                        )  # (num_agents*batch_size, obs_shape)
                        q_vals, updates = network.apply(
                            {"params": params, "batch_stats": train_state.batch_stats},
                            agent_in,
                            train=True,
                            mutable=["batch_stats"],
                        )  # (num_agents*batch_size, num_actions)
                        q_vals = q_vals.reshape(
                            env.num_agents, -1, wrapped_env.max_action_space
                        )  # (num_agents, batch_size, num_actions)

                        chosen_action_qvals = jnp.take_along_axis(
                            q_vals,
                            jnp.expand_dims(minibatch.action, axis=-1),
                            axis=-1,
                        ).squeeze(
                            axis=-1
                        )  # (num_agents, batch_size,)
                        vdn_chosen_action_qvals = jnp.sum(
                            chosen_action_qvals, axis=0
                        )  # (batch_size)

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
                    # x: (num_steps, num_agents, num_envs, ...)
                    x = jnp.swapaxes(x, 1, 2)  # num_steps, num_envs, num_agents...
                    x = x.reshape(-1, *x.shape[2:])  # num_steps*num_envs, num_agents
                    x = jax.random.permutation(
                        rng, x, axis=0
                    )  # shuffle the transitions
                    x = x.reshape(
                        config["NUM_MINIBATCHES"], -1, *x.shape[1:]
                    )  # num_minibatches, num_envs/num_minbatches, num_agents, ...
                    x = jnp.swapaxes(
                        x, 1, 2
                    )  # num_minibatches, num_agents, num_envs/num_minbatches, ...
                    return x

                rng, _rng = jax.random.split(rng)
                minibatches = jax.tree.map(
                    lambda x: preprocess_transition(x, _rng),
                    transitions,
                )  # num_minibatches, num_agents, num_envs/num_minbatches ...
                targets = jax.random.permutation(_rng, lambda_targets)
                targets = targets.reshape(config["NUM_MINIBATCHES"], -1)

                rng, _rng = jax.random.split(rng)
                (train_state, rng), (loss, qvals) = jax.lax.scan(
                    _learn_phase, (train_state, rng), (minibatches, targets)
                )

                return (train_state, rng), (loss, qvals)

            rng, _rng = jax.random.split(rng)
            (train_state, rng), (loss, qvals) = jax.lax.scan(
                _learn_epoch, (train_state, rng), None, config["NUM_EPOCHS"]
            )

            train_state = train_state.replace(n_updates=train_state.n_updates + 1)
            metrics = {
                "env_step": train_state.timesteps,
                "update_steps": train_state.n_updates,
                "grad_steps": train_state.grad_steps,
                "loss": loss.mean(),
                "qvals": qvals.mean(),
            }
            metrics.update(jax.tree.map(lambda x: x.mean(), infos))

            if config.get("TEST_DURING_TRAINING", True):
                rng, _rng = jax.random.split(rng)
                test_state = jax.lax.cond(
                    train_state.n_updates
                    % int(config["NUM_UPDATES"] * config["TEST_INTERVAL"])
                    == 0,
                    lambda _: get_greedy_metrics(_rng, train_state),
                    lambda _: test_state,
                    operand=None,
                )
                metrics.update({"test_" + k: v for k, v in test_state.items()})

            # report on wandb if required
            if config["WANDB_MODE"] != "disabled":

                def callback(metrics, original_seed):
                    if config.get("WANDB_LOG_ALL_SEEDS", False):
                        metrics.update(
                            {
                                f"rng{int(original_seed)}/{k}": v
                                for k, v in metrics.items()
                            }
                        )
                    wandb.log(metrics, step=metrics["update_steps"])

                jax.debug.callback(callback, metrics, original_seed)

            runner_state = (train_state, tuple(expl_state), test_state, rng)

            return runner_state, metrics

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
                _greedy_env_step, step_state, None, config["TEST_NUM_STEPS"]
            )
            metrics = {
                "returned_episode_returns": jnp.nanmean(
                    jnp.where(
                        infos["returned_episode"],
                        infos["returned_episode_returns"],
                        jnp.nan,
                    )
                )
            }
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


def env_from_config(config):
    env_name = config["ENV_NAME"]
    # smax init neeeds a scenario
    if "smax" in env_name.lower():
        config["ENV_KWARGS"]["scenario"] = map_name_to_scenario(config["MAP_NAME"])
        env_name = f"{config['ENV_NAME']}_{config['MAP_NAME']}"
        env = make(config["ENV_NAME"], **config["ENV_KWARGS"])
        env = SMAXLogWrapper(env)
    # overcooked needs a layout
    elif "overcooked" in env_name.lower():
        env_name = f"{config['ENV_NAME']}_{config['ENV_KWARGS']['layout']}"
        config["ENV_KWARGS"]["layout"] = overcooked_layouts[
            config["ENV_KWARGS"]["layout"]
        ]
        env = make(config["ENV_NAME"], **config["ENV_KWARGS"])
        env = LogWrapper(env)
    elif "mpe" in env_name.lower():
        env = make(config["ENV_NAME"], **config["ENV_KWARGS"])
        env = MPELogWrapper(env)
    else:
        env = make(config["ENV_NAME"], **config["ENV_KWARGS"])
        env = LogWrapper(env)
    return env, env_name


def single_run(config):

    config = {**config, **config["alg"]}  # merge the alg config with the main config
    print("Config:\n", OmegaConf.to_yaml(config))

    alg_name = config.get('ALG_NAME', "pqn_vdn_cnn")
    env, env_name = env_from_config(copy.deepcopy(config))

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
    train_vjit = jax.jit(jax.vmap(make_train(config, env)))
    outs = jax.block_until_ready(train_vjit(rngs))

    # save params
    if config.get("SAVE_PATH", None) is not None:
        from jaxmarl.wrappers.baselines import save_params

        model_state = outs["runner_state"][0]
        save_dir = os.path.join(config["SAVE_PATH"], env_name)
        os.makedirs(save_dir, exist_ok=True)
        OmegaConf.save(
            config,
            os.path.join(
                save_dir, f'{alg_name}_{env_name}_seed{config["SEED"]}_config.yaml'
            ),
        )

        for i, rng in enumerate(rngs):
            params = jax.tree.map(lambda x: x[i], model_state.params)
            save_path = os.path.join(
                save_dir,
                f'{alg_name}_{env_name}_seed{config["SEED"]}_vmap{i}.safetensors',
            )
            save_params(params, save_path)


def tune(default_config):
    """Hyperparameter sweep with wandb."""

    default_config = {
        **default_config,
        **default_config["alg"],
    }  # merge the alg config with the main config
    env_name = default_config["ENV_NAME"]
    alg_name = "pqn_vdn_cnn"
    env, env_name = env_from_config(default_config)

    def wrapped_make_train():
        wandb.init(project=default_config["PROJECT"])

        # update the default params
        config = copy.deepcopy(default_config)
        for k, v in dict(wandb.config).items():
            config[k] = v

        print("running experiment with params:", config)

        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, config["NUM_SEEDS"])
        train_vjit = jax.jit(jax.vmap(make_train(config, env)))
        outs = jax.block_until_ready(train_vjit(rngs))

    sweep_config = {
        "name": f"{alg_name}_{env_name}",
        "method": "bayes",
        "metric": {
            "name": "test_returned_episode_returns",
            "goal": "maximize",
        },
        "parameters": {
            "LR": {
                "values": [
                    0.001,
                    0.00075,
                    0.0005,
                    0.00025,
                    0.0001,
                    0.000075,
                    0.00005,
                ]
            },
            "NUM_ENVS": {"values": [8, 32, 64, 128]},
            "NUM_STEPS": {"values": [8, 16, 32, 64, 128]},
            "LAMBDA": {"values": [0, 0.3, 0.5, 0.7, 0.9]},
            "EPS_FINISH": {"values": [0.01, 0.2, 0.1]},
            "NUM_MINIBATCHES": {"values": [2, 4, 8, 16]},
            "NUM_EPOCHS": {"values": [1, 2, 4]},
        },
    }

    wandb.login()
    sweep_id = wandb.sweep(
        sweep_config, entity=default_config["ENTITY"], project=default_config["PROJECT"]
    )
    wandb.agent("11ecmms5", wrapped_make_train, count=300)


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
