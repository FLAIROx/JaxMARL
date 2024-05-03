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


class ScannedRNN(nn.Module):

    @partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        hidden_size = rnn_state.shape[-1]
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(hidden_size, *resets.shape),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(hidden_size)(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(hidden_size, *batch_size):
        # Use a dummy key since the default state init fn is just zeros.
        return nn.GRUCell(hidden_size, parent=None).initialize_carry(
            jax.random.PRNGKey(0), (*batch_size, hidden_size)
        )


class QNetwork(nn.Module):
    action_dim: int
    hidden_size: int = 512
    n_layers: int = 4
    norm_type: str = "layer_norm"
    init_scale: float = 1.0
    dueling: bool = False

    @nn.compact
    def __call__(self, hidden, x, dones, train: bool = False):
        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        elif self.norm_type == "batch_norm":
            normalize = lambda x: nn.BatchNorm(use_running_average=not train)(x)
            x = nn.BatchNorm(use_running_average=not train)(x)  # normalize input
        else:
            normalize = lambda x: x

        if self.norm_type != "batch_norm":
            # dummy normalize input if not using batch norm just for global compatibility
            x_dummy = nn.BatchNorm(use_running_average=not train)(x)

        for l in range(self.n_layers):
            x = nn.Dense(self.hidden_size)(x)
            x = normalize(x)
            x = nn.relu(x)

        rnn_in = (x, dones)
        hidden, x = ScannedRNN()(hidden, rnn_in)
        x = normalize(x)

        if self.dueling:
            adv = nn.Dense(self.action_dim)(x)
            val = nn.Dense(1)(x)
            q_vals = val + adv - jnp.mean(adv, axis=-1, keepdims=True)
        else:
            q_vals = nn.Dense(self.action_dim)(x)

        return hidden, q_vals


@chex.dataclass(frozen=True)
class Transition:
    last_hs: chex.Array
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    last_done: chex.Array
    avail_actions: chex.Array


class CustomTrainState(TrainState):
    batch_stats: Any
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0


def make_train(config, env):

    assert (
        config["NUM_ENVS"] % config["NUM_MINIBATCHES"] == 0
    ), "NUM_ENVS must be divisible by NUM_MINIBATCHES"

    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    eps_scheduler = optax.linear_schedule(
        config["EPS_START"],
        config["EPS_FINISH"],
        config["EPS_DECAY"] * config["NUM_UPDATES"],
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
            dueling=config.get("DUELING", False),
        )

        def create_agent(rng):
            init_x = (
                jnp.zeros(
                    (1, 1, wrapped_env.obs_size)
                ),  # (time_step, batch_size, obs_size)
                jnp.zeros((1, 1)),  # (time_step, batch size)
            )
            init_hs = ScannedRNN.initialize_carry(
                config["HIDDEN_SIZE"], 1
            )  # (batch_size, hidden_dim)
            network_variables = network.init(rng, init_hs, *init_x, train=False)

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

            train_state, memory_transitions, expl_state, test_state, rng = runner_state

            # SAMPLE PHASE
            def _step_env(carry, _):
                expl_state, rng = carry
                hs, last_obs, last_dones, env_state = expl_state
                rng, rng_a, rng_s = jax.random.split(rng, 3)

                _obs = batchify(last_obs)[
                    :, np.newaxis
                ]  # (num_agents, 1 (dummy time), num_envs, obs_size)
                _dones = batchify(last_dones)[
                    :, np.newaxis
                ]  # (num_agents, 1 (dummy time), num_envs, obs_size)
                new_hs, q_vals = jax.vmap(network.apply, in_axes=(None, 0, 0, 0, None))(
                    {
                        "params": train_state.params,
                        "batch_stats": train_state.batch_stats,
                    },
                    hs,
                    _obs,
                    _dones,
                    False,
                )  # (num_agents, 1, num_envs, num_actions)
                q_vals = q_vals.squeeze(
                    axis=1
                )  # (num_agents, num_envs, num_actions) remove the time dim

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

                transition = Transition(
                    last_hs=hs,  # (num_agents, num_envs, hidden_size)
                    obs=batchify(last_obs),  # (num_agents, num_envs, obs_shape)
                    action=batchify(new_action),  # (num_agents, num_envs,)
                    reward=reward["__all__"][np.newaxis],  # (1, num_envs,)
                    done=new_done["__all__"][np.newaxis],  # (1, num_envs,)
                    last_done=batchify(last_dones),  # (num_agents, num_envs,)
                    avail_actions=batchify(
                        avail_actions
                    ),  # (num_agents, num_envs, num_actions)
                )
                return ((new_hs, new_obs, new_done, new_env_state), rng), (
                    transition,
                    info,
                )

            # step the env
            rng, _rng = jax.random.split(rng)
            (expl_state, rng), (transitions, infos) = jax.lax.scan(
                _step_env,
                (expl_state, _rng),
                None,
                config["NUM_STEPS"],
            )

            train_state = train_state.replace(
                timesteps=train_state.timesteps
                + config["NUM_STEPS"] * config["NUM_ENVS"]
            )  # update timesteps count

            # insert the transitions into the memory
            memory_transitions = jax.tree_map(
                lambda x, y: jnp.concatenate([x[config["NUM_STEPS"] :], y], axis=0),
                memory_transitions,
                transitions,
            )

            # NETWORKS UPDATE
            def _learn_epoch(carry, _):
                train_state, rng = carry

                def _learn_phase(carry, minibatch):

                    # minibatch shape: num_steps, num_agents, batch_size, ...
                    # with batch_size = num_envs/num_minibatches

                    train_state, rng = carry
                    hs = minibatch.last_hs[0].reshape(
                        -1, config["HIDDEN_SIZE"]
                    )  # hs of oldest step (num_agents, batch_size, hidden_size)
                    agent_in = (
                        minibatch.obs,
                        minibatch.last_done,
                    )
                    # batchify the agent input: num_agents*batch_size
                    agent_in = jax.tree_util.tree_map(
                        lambda x: x.reshape(x.shape[0], -1, *x.shape[3:]), agent_in
                    )  # (num_steps, num_agents*batch_size, ...)
                    

                    def _loss_fn(params):
                        (_, q_vals), updates = partial(network.apply, train=True,mutable=["batch_stats"])(
                            {"params": params, "batch_stats": train_state.batch_stats},
                            hs,
                            *agent_in,
                        )  # (num_steps, num_agents*batch_size, num_actions)
                        q_vals = q_vals.reshape(q_vals.shape[0], env.num_agents, -1, q_vals.shape[-1])  # (num_steps, num_agents, batch_size, num_actions)

                        chosen_action_qvals = jnp.take_along_axis(
                            q_vals,
                            jnp.expand_dims(minibatch.action, axis=-1),
                            axis=-1,
                        ).squeeze(
                            axis=-1
                        )  # (num_steps, num_agents, batch_size,)
                        vdn_chosen_action_qvals = jnp.sum(chosen_action_qvals, axis=1)[
                            :-1
                        ]  # (num_steps-1, batch_size,)

                        q_next = jax.lax.stop_gradient(q_vals)
                        unavailable_actions = 1 - minibatch.avail_actions
                        valid_q_vals = q_next - (unavailable_actions * 1e10)
                        q_next = jnp.max(
                            valid_q_vals, axis=-1
                        )  # (num_steps, num_agents, batch_size,)

                        vdn_q_next = jnp.sum(q_next, axis=1)  # (num_steps, batch_size,)
                        target = (
                            minibatch.reward[
                                :-1, 0
                            ]  # [:-1, 0 is __all__, is all but the last step]
                            + (1 - minibatch.done[:-1, 0])
                            * config["GAMMA"]
                            * vdn_q_next[1:]
                        )  # (num_steps-1, batch_size,)

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
                    x = jax.random.permutation(rng, x, axis=2)  # shuffle the transitions
                    x = x.reshape(
                        *x.shape[:2], config["NUM_MINIBATCHES"], -1, *x.shape[3:]
                    )  # num_steps, num_agents, minibatches, batch_size/num_minbatches,
                    new_order = [2, 0, 1, 3] + list(
                        range(4, x.ndim)
                    )  # (minibatches, num_steps, num_agents, batch_size/num_minbatches, ...)
                    x = jnp.transpose(x, new_order)
                    return x

                rng, _rng = jax.random.split(rng)
                minibatches = jax.tree_util.tree_map(
                    lambda x: preprocess_transition(x, _rng),
                    memory_transitions,
                )  # num_minibatches, num_steps+memory_window, num_agents, batch_size/num_minbatches, num_agents, ...

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
                "update_steps": train_state.n_updates,
                "grad_steps": train_state.grad_steps,
                "loss": loss.mean(),
                "qvals": qvals.mean(),
                "epsilon": eps_scheduler(train_state.n_updates),
            }
            metrics.update(
                jax.tree_map(
                    lambda x: jnp.nanmean(
                        jnp.where(
                            infos["returned_episode"],
                            x,
                            jnp.nan,
                        )
                    ),
                    infos,
                )
            )

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
                    wandb.log(metrics)

                jax.debug.callback(callback, metrics)

            runner_state = (
                train_state,
                memory_transitions,
                expl_state,
                test_state,
                rng,
            )

            return runner_state, None

        def get_greedy_metrics(rng, train_state):
            if not config.get("TEST_DURING_TRAINING", True):
                return None
            """Help function to test greedy policy during training"""

            def _greedy_env_step(step_state, unused):
                env_state, last_obs, last_dones, hstate, rng = step_state
                rng, key_s = jax.random.split(rng)
                _obs = batchify(last_obs)[:, np.newaxis]
                _dones = batchify(last_dones)[:, np.newaxis]
                hstate, q_vals = jax.vmap(
                    partial(network.apply), in_axes=(None, 0, 0, 0, None)
                )(
                    {
                        "params": train_state.params,
                        "batch_stats": train_state.batch_stats,
                    },
                    hstate,
                    _obs,
                    _dones,
                    False,
                )
                q_vals = q_vals.squeeze(axis=1)
                valid_actions = wrapped_env.get_valid_actions(env_state.env_state)
                actions = get_greedy_actions(q_vals, batchify(valid_actions))
                actions = unbatchify(actions)
                obs, env_state, rewards, dones, infos = test_env.batch_step(
                    key_s, env_state, actions
                )
                step_state = (env_state, obs, dones, hstate, rng)
                return step_state, (rewards, dones, infos)

            rng, _rng = jax.random.split(rng)
            init_obs, env_state = test_env.batch_reset(_rng)
            init_dones = {
                agent: jnp.zeros((config["NUM_TEST_EPISODES"]), dtype=bool)
                for agent in env.agents + ["__all__"]
            }
            rng, _rng = jax.random.split(rng)
            hstate = ScannedRNN.initialize_carry(
                config["HIDDEN_SIZE"], len(env.agents), config["NUM_TEST_EPISODES"]
            )  # (n_agents*n_envs, hs_size)
            step_state = (
                env_state,
                init_obs,
                init_dones,
                hstate,
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
        init_dones = {
            agent: jnp.zeros((config["NUM_ENVS"]), dtype=bool)
            for agent in env.agents + ["__all__"]
        }
        init_hs = ScannedRNN.initialize_carry(
            config["HIDDEN_SIZE"], len(env.agents), config["NUM_ENVS"]
        )
        expl_state = (init_hs, obs, init_dones, env_state)

        # step randomly once to have the initial memory window
        def _random_step(carry, _):
            expl_state, rng = carry
            hs, last_obs, last_dones, env_state = expl_state
            rng, rng_a, rng_s = jax.random.split(rng, 3)
            _obs = batchify(last_obs)[:, np.newaxis]
            _dones = batchify(last_dones)[:, np.newaxis]
            avail_actions = wrapped_env.get_valid_actions(env_state.env_state)
            new_hs, q_vals = jax.vmap(network.apply, in_axes=(None, 0, 0, 0, None))(
                {
                    "params": train_state.params,
                    "batch_stats": train_state.batch_stats,
                },
                hs,
                _obs,
                _dones,
                False,
            )  # (num_agents, 1, num_envs, num_actions)
            _rngs = jax.random.split(rng_a, env.num_agents)
            new_action = {
                agent: wrapped_env.batch_sample(_rngs[i], agent)
                for i, agent in enumerate(env.agents)
            }
            new_obs, new_env_state, reward, new_done, info = wrapped_env.batch_step(
                rng_s, env_state, new_action
            )
            transition = Transition(
                last_hs=hs,  # (num_agents, num_envs, hidden_size)
                obs=batchify(last_obs),  # (num_agents, num_envs, obs_shape)
                action=batchify(new_action),  # (num_agents, num_envs,)
                reward=reward["__all__"][np.newaxis],  # (1, num_envs,)
                done=new_done["__all__"][np.newaxis],  # (1, num_envs,)
                last_done=batchify(last_dones),  # (1, num_envs,)
                avail_actions=batchify(
                    avail_actions
                ),  # (num_agents, num_envs, num_actions)
            )
            return ((new_hs, new_obs, new_done, new_env_state), rng), transition

        rng, _rng = jax.random.split(rng)
        (expl_state, rng), memory_transitions = jax.lax.scan(
            _random_step,
            (expl_state, _rng),
            None,
            config["MEMORY_WINDOW"] + config["NUM_STEPS"],
        )

        # train
        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, memory_transitions, expl_state, test_state, _rng)

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
    # overcooked needs a layout
    elif "overcooked" in env_name.lower():
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
        outs = jax.jit(make_train(config["alg"], env))(rng)


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
        "name": "pqn_smax",
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
            "LR_DECAY": {"values": [True, False]},
            "NORM_TYPE": {"values": ["none", "batch_norm", "layer_norm"]},
            "HIDDEN_SIZE": {"values": [256, 512, 1024]},
            "N_LAYERS": {"values": [1,2,3]},
            "EPS_FINISH": {"values": [0.01, 0.05, 0.005, 0.001]},
            "MAX_GRAD_NORM": {"values": [0.5, 1, 10, 20]},
            "NUM_MINIBATCHES": {"values": [1, 2, 4, 8, 16]},
            "NUM_EPOCHS": {"values": [1, 2, 3, 4]},
            "NUM_ENVS": {"values": [32, 64, 512, 1024]},
            "MEMORY_WINDOW": {"values": [0, 4, 16, 32, 64, 128]},
            "NUM_STEPS": {"values": [0, 4, 16, 32, 64, 128]},
        },
    }

    wandb.login()
    sweep_id = wandb.sweep(
        sweep_config, entity=default_config["ENTITY"], project=default_config["PROJECT"]
    )
    wandb.agent(sweep_id, wrapped_make_train, count=500)


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