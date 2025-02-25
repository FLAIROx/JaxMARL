import os
import copy
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Tuple, Union, Dict
import wandb
import functools
from flax.training.train_state import TrainState
import distrax
import hydra
from omegaconf import DictConfig, OmegaConf
from functools import partial

from jaxmarl import make
from jaxmarl.wrappers.baselines import LogWrapper, SMAXLogWrapper, JaxMARLWrapper
from jaxmarl.viz.utracking_visualizer import animate_from_infos
from jaxmarl.wrappers.baselines import save_params, load_params


class ScannedRNN(nn.Module):
    @functools.partial(
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
        # print('ins', ins)
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], ins.shape[1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size, parent=None)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class ActorRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):

        obs, dones, avail_actions = x

        embedding = obs 
        for l in range(self.config["NUM_LAYERS"]):
            embedding = nn.Dense(
                self.config["FC_DIM_SIZE"],
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(embedding)
            embedding = nn.LayerNorm()(embedding)
            embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(
            self.config["GRU_HIDDEN_DIM"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        actor_mean = nn.LayerNorm()(actor_mean)
        actor_mean = nn.relu(actor_mean)

        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        unavail_actions = 1 - avail_actions
        action_logits = actor_mean - (unavail_actions * 1e10)

        pi = distrax.Categorical(logits=action_logits)

        return hidden, pi


class CriticRNN(nn.Module):
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        world_state, dones = x

        embedding = world_state 
        for l in range(self.config["NUM_LAYERS"]):
            embedding = nn.Dense(
                self.config["FC_DIM_SIZE"],
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(embedding)
            embedding = nn.LayerNorm()(embedding)
            embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)

        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        critic = nn.Dense(
            self.config["GRU_HIDDEN_DIM"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        critic = nn.LayerNorm()(critic)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    world_state: jnp.ndarray
    info: jnp.ndarray
    avail_actions: jnp.ndarray


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    # print('batchify', x.shape)
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def make_train(config):

    config["ENV_KWARGS"]["infos_for_render"] = config["ENV_KWARGS"].get(
        "infos_for_render", config.get("ANIMATION_LOG_INTERVAL", None) is not None
    )
    env = make(config["ENV_NAME"], **config["ENV_KWARGS"])
    print("env created")

    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    config["CLIP_EPS"] = (
        config["CLIP_EPS"] / env.num_agents
        if config["SCALE_CLIP_EPS"]
        else config["CLIP_EPS"]
    )

    env = LogWrapper(env)

    if config["LOAD_PATH"] is not None:
        config["MODEL_PARAMS"] = load_params(config["LOAD_PATH"])
        print("loaded model from", config["LOAD_PATH"])

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        original_seed = rng[0]
        # INIT NETWORK
        actor_network = ActorRNN(env.action_space(env.agents[0]).n, config=config)
        critic_network = CriticRNN(config=config)
        rng, _rng_actor, _rng_critic = jax.random.split(rng, 3)

        if config["LOAD_PATH"] is not None:
            actor_network_params = config["MODEL_PARAMS"]["actor"]
            critic_network_params = config["MODEL_PARAMS"]["critic"]
        else:
            ac_init_x = (
                jnp.zeros(
                    (1, config["NUM_ENVS"], env.observation_space(env.agents[0]).shape[0])
                ),
                jnp.zeros((1, config["NUM_ENVS"])),
                jnp.zeros((1, config["NUM_ENVS"], env.action_space(env.agents[0]).n)),
            )
            ac_init_hstate = ScannedRNN.initialize_carry(
                config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]
            )
            actor_network_params = actor_network.init(_rng_actor, ac_init_hstate, ac_init_x)

        if config["LOAD_PATH"] is None or not config["LOAD_CRITIC"]:
            cr_init_x = (
                jnp.zeros(
                    (
                        1,
                        config["NUM_ENVS"],
                        env.world_state_space.shape[0],
                    )
                ),
                jnp.zeros((1, config["NUM_ENVS"])),
            )
            cr_init_hstate = ScannedRNN.initialize_carry(
                config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]
            )
            critic_network_params = critic_network.init(
                _rng_critic, cr_init_hstate, cr_init_x
            )

        if config["ANNEAL_LR"]:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        actor_train_state = TrainState.create(
            apply_fn=actor_network.apply,
            params=actor_network_params,
            tx=actor_tx,
        )
        critic_train_state = TrainState.create(
            apply_fn=actor_network.apply,
            params=critic_network_params,
            tx=critic_tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        ac_init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"]
        )
        cr_init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"]
        )

        # TRAIN LOOP
        def _update_step(update_runner_state, unused):
            # COLLECT TRAJECTORIES
            runner_state, update_steps = update_runner_state

            def _env_step(runner_state, unused):
                train_states, env_state, last_obs, last_done, hstates, rng = (
                    runner_state
                )

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                avail_actions = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(
                    batchify(avail_actions, env.agents, config["NUM_ACTORS"])
                )
                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                print("obs shape:", obs_batch.shape)
                ac_in = (
                    obs_batch[np.newaxis, :],
                    last_done[np.newaxis, :],
                    avail_actions,
                )
                # print('env step ac in', ac_in)
                ac_hstate, pi = actor_network.apply(
                    train_states[0].params, hstates[0], ac_in
                )
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(
                    action, env.agents, config["NUM_ENVS"], env.num_agents
                )
                env_act = {k: v.squeeze() for k, v in env_act.items()}

                # VALUE
                world_state = last_obs["world_state"]  # num_envs, state_size
                world_state = jnp.repeat(
                    world_state, env.num_agents, axis=0
                )  # repeat world_state for each agent
                world_state = world_state.reshape(
                    (config["NUM_ACTORS"], -1)
                )  # (num_actors, world_state_size)

                cr_in = (
                    world_state[None, :],
                    last_done[np.newaxis, :],
                )
                cr_hstate, value = critic_network.apply(
                    train_states[1].params, hstates[1], cr_in
                )

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)
                # info = jax.tree_map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                done_batch = batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()
                transition = Transition(
                    jnp.tile(done["__all__"], env.num_agents),
                    last_done,
                    action.squeeze(),
                    value.squeeze(),
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob.squeeze(),
                    obs_batch,
                    world_state,
                    info,
                    avail_actions,
                )
                runner_state = (
                    train_states,
                    env_state,
                    obsv,
                    done_batch,
                    (ac_hstate, cr_hstate),
                    rng,
                )
                return runner_state, transition

            initial_hstates = runner_state[-2]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE
            train_states, env_state, last_obs, last_done, hstates, rng = runner_state

            last_world_state = last_obs["world_state"]  # num_envs, state_size
            last_world_state = jnp.repeat(
                last_world_state, env.num_agents, axis=0
            )  # repeat world_state for each agent
            last_world_state = last_world_state.reshape(
                (config["NUM_ACTORS"], -1)
            )  # (num_actors, world_state_size)

            cr_in = (
                last_world_state[None, :],
                last_done[np.newaxis, :],
            )
            _, last_val = critic_network.apply(
                train_states[1].params, hstates[1], cr_in
            )
            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.global_done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_states, batch_info):
                    actor_train_state, critic_train_state = train_states
                    ac_init_hstate, cr_init_hstate, traj_batch, advantages, targets = (
                        batch_info
                    )

                    def _actor_loss_fn(actor_params, init_hstate, traj_batch, gae):
                        # RERUN NETWORK
                        _, pi = actor_network.apply(
                            actor_params,
                            init_hstate.squeeze(),
                            (traj_batch.obs, traj_batch.done, traj_batch.avail_actions),
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE ACTOR LOSS
                        logratio = log_prob - traj_batch.log_prob
                        ratio = jnp.exp(logratio)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        # debug
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["CLIP_EPS"])

                        actor_loss = (
                            loss_actor
                            - config["ENT_COEF"] * entropy
                            - config.get("KL_COEF", 0.) * approx_kl
                        )

                        return actor_loss, (
                            loss_actor,
                            entropy,
                            ratio,
                            approx_kl,
                            clip_frac,
                        )

                    def _critic_loss_fn(
                        critic_params, init_hstate, traj_batch, targets
                    ):
                        # RERUN NETWORK
                        _, value = critic_network.apply(
                            critic_params,
                            init_hstate.squeeze(),
                            (traj_batch.world_state, traj_batch.done),
                        )

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )
                        critic_loss = config["VF_COEF"] * value_loss
                        return critic_loss, (value_loss)

                    actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                    actor_loss, actor_grads = actor_grad_fn(
                        actor_train_state.params, ac_init_hstate, traj_batch, advantages
                    )
                    critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                    critic_loss, critic_grads = critic_grad_fn(
                        critic_train_state.params, cr_init_hstate, traj_batch, targets
                    )

                    actor_train_state = actor_train_state.apply_gradients(
                        grads=actor_grads
                    )
                    critic_train_state = critic_train_state.apply_gradients(
                        grads=critic_grads
                    )

                    total_loss = actor_loss[0] + critic_loss[0]
                    loss_info = {
                        "total_loss": total_loss,
                        "actor_loss": actor_loss[0],
                        "value_loss": critic_loss[0],
                        "entropy": actor_loss[1][1],
                        "ratio": actor_loss[1][2],
                        "approx_kl": actor_loss[1][3],
                        "clip_frac": actor_loss[1][4],
                    }

                    return (actor_train_state, critic_train_state), loss_info

                (
                    train_states,
                    init_hstates,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state
                rng, _rng = jax.random.split(rng)

                init_hstates = jax.tree_map(
                    lambda x: jnp.reshape(x, (1, config["NUM_ACTORS"], -1)),
                    init_hstates,
                )

                batch = (
                    init_hstates[0],
                    init_hstates[1],
                    traj_batch,
                    advantages.squeeze(),
                    targets.squeeze(),
                )
                permutation = jax.random.permutation(_rng, config["NUM_ACTORS"])

                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )

                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["NUM_MINIBATCHES"], -1]
                            + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )

                # train_states = (actor_train_state, critic_train_state)
                train_states, loss_info = jax.lax.scan(
                    _update_minbatch, train_states, minibatches
                )
                update_state = (
                    train_states,
                    jax.tree_map(lambda x: x.squeeze(), init_hstates),
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, loss_info

            update_state = (
                train_states,
                initial_hstates,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            loss_info["ratio_0"] = loss_info["ratio"].at[0, 0].get()

            train_states = update_state[0]
            infos = traj_batch.info
            rng = update_state[-1]

            metrics = {
                "env_step": update_steps * config["NUM_ENVS"] * config["NUM_STEPS"],
                "update_steps": update_steps,
            }
            metrics.update(jax.tree_map(lambda x: x.mean(), infos))
            metrics.update(jax.tree_map(lambda x: x.mean(), loss_info))

            # report on wandb if required
            if config["WANDB_MODE"] != "disabled":

                def callback(metrics, original_seed, render_infos=None, model_state=None):
                    if config.get("WANDB_LOG_ALL_SEEDS", False):
                        metrics.update(
                            {
                                f"rng{int(original_seed)}/{k}": v
                                for k, v in metrics.items()
                            }
                        )

                    # log with animation
                    if (
                        config.get("ANIMATION_LOG_INTERVAL", None) is not None
                        and metrics["update_steps"]
                        % int(config["NUM_UPDATES"] * config["ANIMATION_LOG_INTERVAL"])
                        == 0
                    ):
                        animation_path = os.path.join(
                            wandb.run.dir, f"animation_step{int(metrics['update_steps'])}"
                        )
                        animate_from_infos(
                            render_infos,
                            num_agents=config["ENV_KWARGS"]["num_agents"],
                            save_path=animation_path,
                        )

                        wandb.log(
                            {
                                **metrics,
                                "animation": wandb.Video(
                                    animation_path + ".gif",
                                    caption=f"step{int(metrics['update_steps'])}",
                                    fps=10,
                                    format="gif",
                                )
                            },
                            step=metrics["update_steps"],
                        )


                    else:
                        # log without animation
                        wandb.log(metrics, step=metrics["update_steps"])

                    # save params
                    if (
                        config.get("CHECKPOINT_INTERVAL", None) is not None
                        and metrics["update_steps"]
                        % int(config["NUM_UPDATES"] * config["CHECKPOINT_INTERVAL"])
                        == 0
                    ):
                        
                        env_name = f'utracking_{config["ENV_KWARGS"]["num_agents"]}_vs_{config["ENV_KWARGS"]["num_landmarks"]}'
                        alg_name = config.get("ALG_NAME", "mappo_rnn_utracking")
                        
                        print('Saving Checkpoint')
    
                        model_state = {"actor": model_state[0].params, "critic": model_state[1].params}
                        save_dir = os.path.join(config["SAVE_PATH"], env_name, alg_name)
                        os.makedirs(save_dir, exist_ok=True)

                        save_path = os.path.join(
                            save_dir,
                            f"{alg_name}_{env_name}_step{int(metrics['update_steps'])}_rng{int(original_seed)}.safetensors",
                        )
                        save_params(model_state, save_path)

                if config.get("ANIMATION_LOG_INTERVAL", None) is not None:

                    def get_complete_rollout(rng, params):

                        rng, _rng = jax.random.split(rng)
                        init_obs, init_env_state = env.reset(_rng)
                        init_dones = jnp.zeros((env.num_agents), dtype=bool)

                        init_hstate = ScannedRNN.initialize_carry(
                            env.num_agents, config["GRU_HIDDEN_DIM"]
                        )

                        def step_agent(rng, hstate, obsv, last_done, env_state):

                            avail_actions = env.get_avail_actions(env_state.env_state)
                            obs_batch = batchify(obsv, env.agents, env.num_agents)
                            avail_actions = batchify(
                                avail_actions, env.agents, env.num_agents
                            )
                            ac_in = (
                                obs_batch[np.newaxis, :],
                                last_done[np.newaxis, :],
                                avail_actions,
                            )
                            new_hstate, pi = actor_network.apply(params, hstate, ac_in)
                            action = pi.sample(seed=rng)
                            env_act = unbatchify(action, env.agents, 1, env.num_agents)
                            env_act = {k: v.squeeze() for k, v in env_act.items()}
                            return new_hstate, env_act

                        def env_step(carry, _):

                            rng, hstate, obsv, last_done, env_state = carry

                            rng, _rng = jax.random.split(rng)
                            new_hstate, env_act = step_agent(
                                _rng, hstate, obsv, last_done, env_state
                            )

                            rng, _rng = jax.random.split(rng)
                            new_obsv, new_env_state, reward, done, info = env.step(
                                _rng, env_state, env_act
                            )

                            # TODO: check which dimension is squeezed, gives problem with 1 agent
                            new_last_done = batchify(
                                done, env.agents, env.num_agents
                            ).squeeze(-1)

                            return (
                                rng,
                                new_hstate,
                                new_obsv,
                                new_last_done,
                                new_env_state,
                            ), info

                        step_state = (
                            rng,
                            init_hstate,
                            init_obs,
                            init_dones,
                            init_env_state,
                        )

                        step_state, infos = jax.lax.scan(
                            env_step, step_state, None, config["ANIMATION_MAX_STEPS"]
                        )
                        return infos

                    rng, _rng = jax.random.split(rng)
                    render_infos = jax.lax.cond(
                        metrics["update_steps"]
                        % int(config["NUM_UPDATES"] * config["ANIMATION_LOG_INTERVAL"])
                        == 0,
                        lambda _: get_complete_rollout(_rng, train_states[0].params),
                        lambda _: jax.tree_map(
                            lambda x: jnp.repeat(
                                x[0, 0:1], (config["ANIMATION_MAX_STEPS"]), axis=0
                            ),
                            infos,  # dummy infos placeholder ensuring the same shape of single_rollout
                        ),
                        operand=None,
                    )
                else:
                    render_infos = None

                jax.debug.callback(callback, metrics, original_seed, render_infos, train_states)

            update_steps = update_steps + 1
            runner_state = (train_states, env_state, last_obs, last_done, hstates, rng)
            return (runner_state, update_steps), metrics

        rng, _rng = jax.random.split(rng)
        runner_state = (
            (actor_train_state, critic_train_state),
            env_state,
            obsv,
            jnp.zeros((config["NUM_ACTORS"]), dtype=bool),
            (ac_init_hstate, cr_init_hstate),
            _rng,
        )
        (runner_state, update_step), metric = jax.lax.scan(
            _update_step, (runner_state, 0), None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state}

    return train


def single_run(config):

    config = OmegaConf.to_container(config)

    env_name = f'utracking_{config["ENV_KWARGS"]["num_agents"]}_vs_{config["ENV_KWARGS"]["num_landmarks"]}'
    alg_name = config.get("ALG_NAME", "mappo_rnn_utracking")

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["MAPPO", "RNN", "UTRACKING"],
        config=config,
        mode=config["WANDB_MODE"],
        name=f'{alg_name}_{env_name}_seed{config["SEED"]}',
    )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_jit = jax.jit(make_train(copy.deepcopy(config)))
    outs = jax.vmap(train_jit)(rngs)

    if config.get("SAVE_PATH", None) is not None:
        
        model_state = outs["runner_state"][0]
        model_state = {"actor": model_state[0].params, "critic": model_state[1].params}
        save_dir = os.path.join(config["SAVE_PATH"], env_name, alg_name)
        os.makedirs(save_dir, exist_ok=True)
        OmegaConf.save(
            config,
            os.path.join(
                save_dir, f'{alg_name}_{env_name}_seed{config["SEED"]}_config.yaml'
            ),
        )

        for i, rng in enumerate(rngs):
            params = jax.tree_map(lambda x: x[i], model_state)
            save_path = os.path.join(
                save_dir,
                f'{alg_name}_{env_name}_seed{config["SEED"]}_vmap{i}_final.safetensors',
            )
            save_params(params, save_path)


def tune(default_config):
    """Hyperparameter sweep with wandb."""
    import copy

    default_config = OmegaConf.to_container(default_config)

    def wrapped_make_train():

        wandb.init(project=default_config["PROJECT"])
        # update the default params
        config = copy.deepcopy(default_config)
        for k, v in dict(wandb.config).items():
            config[k] = v

        print("running experiment with params:", config)

        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, config["NUM_SEEDS"])
        train_vjit = jax.jit(jax.vmap(make_train(config)))
        outs = jax.block_until_ready(train_vjit(rngs))

    sweep_config = {
        "name": "mappo_rnn_utracking",
        "method": "bayes",
        "metric": {
            "name": "returned_episode_returns",
            "goal": "maximize",
        },
        "parameters": {
            # "NUM_ENVS": {"values": [32, 64, 128, 256]},
            "LR": {"values": [0.0005, 0.0001, 0.00005, 0.00001]},
            # "ACTIVATION": {"values": ["relu", "tanh"]},
            # "UPDATE_EPOCHS": {"values": [2, 4, 8]},
            "NUM_MINIBATCHES": {"values": [2, 4, 8, 16]},
            "CLIP_EPS": {"values": [0.1, 0.2, 0.3]},
            # "ENT_COEF": {"values": [0.0001, 0.001, 0.01]},
            "NUM_STEPS": {"values": [32, 64, 128]},
            "GAMMA": {"values": [0.99, 0.999, 0.9]},
            "GAE_LAMBDA": {"values": [0.95, 0.99, 0.9]},
            "VF_COEF": {"values": [0.1, 0.5, 1.0]},
        },
    }

    wandb.login()
    sweep_id = wandb.sweep(
        sweep_config, entity=default_config["ENTITY"], project=default_config["PROJECT"]
    )
    wandb.agent(sweep_id, wrapped_make_train, count=1000)


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="mappo_homogenous_rnn_utracking",
)
def main(config):
    if config["TUNE"]:
        tune(config)
    else:
        single_run(config)


if __name__ == "__main__":
    main()
