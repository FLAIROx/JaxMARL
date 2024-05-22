""" 
Based on PureJaxRL Implementation of PPO
"""

import os
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import distrax
import jaxmarl
from jaxmarl.wrappers.baselines import LogWrapper
import matplotlib.pyplot as plt
import hydra
from omegaconf import OmegaConf
import wandb
import orbax.checkpoint as ocp
from flax.training import checkpoints
from safetensors.flax import save_file
from flax.traverse_util import flatten_dict
from typing import NamedTuple, Dict, Union
from jaxmarl.environments import SimpleFacmacMPE
from jaxmarl.environments.mpe.simple import State, SimpleMPE
from functools import partial
import chex

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
        actor_logtstd = self.param('log_std', nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

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


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def batchify(x: dict, agent_list, num_actors):
    max_dim = max([x[a].shape[-1] for a in agent_list])
    def pad(z, length):
        return jnp.concatenate([z, jnp.zeros(z.shape[:-1] + [length - z.shape[-1]])], -1)

    x = jnp.stack([x[a] if x[a].shape[-1] == max_dim else pad(x[a]) for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def make_train(config):
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_ADVERSARIES"] = env.num_adversaries * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
            config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
            config["NUM_ADVERSARIES"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    # env = FlattenObservationWrapper(env) # NOTE need a batchify wrapper
    env = LogWrapper(env, replace_info=True)

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    def train(rng):

        # INIT NETWORK
        # TODO doesn't work for non-homogenous agents
        network = ActorCritic(env.action_space(env.agents[0]).shape[0], activation=config["ACTIVATION"])
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env.agents[0]).shape)
        network_params = network.init(_rng, init_x)
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(config["LR"], eps=1e-5))

        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset)(reset_rng)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state

                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                # SELECT ACTION
                rng, _rng = jax.random.split(rng)

                pi, value = network.apply(train_state.params, obs_batch)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step)(
                    rng_step, env_state, env_act,
                )

                print("info", info)
                info = jax.tree_map(lambda x: x[:,:env.num_adversaries].reshape((config["NUM_ADVERSARIES"])), info)
                transition = Transition(
                    batchify(done, env.agents[:env.num_adversaries], config["NUM_ADVERSARIES"]).squeeze(),
                    action[:config["NUM_ADVERSARIES"]],
                    value[:config["NUM_ADVERSARIES"]],
                    batchify(reward, env.agents[:env.num_adversaries], config["NUM_ADVERSARIES"]).squeeze(),
                    log_prob[:config["NUM_ADVERSARIES"]],
                    obs_batch[:config["NUM_ADVERSARIES"]],
                    info,
                )
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rng = runner_state
            last_obs_batch = batchify(last_obs, env.agents[:env.num_adversaries], config["NUM_ADVERSARIES"])
            _, last_val = network.apply(train_state.params, last_obs_batch)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
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
                    unroll=8,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                                value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                                0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
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

                        total_loss = (
                                loss_actor
                                + config["VF_COEF"] * value_loss
                                - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                        batch_size == config["NUM_STEPS"] * config["NUM_ADVERSARIES"]
                ), "batch size must be equal to number of steps * number of actors"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss
            
            def callback(metric):
                wandb.log(
                    metric
                )

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]
            
            loss_info = jax.tree_map(lambda x: x.mean(), loss_info)
            # print(metric)
            metric = jax.tree_map(lambda x: x[-2,:].reshape((config["NUM_ENVS"], env.num_adversaries)), metric)
            # print("after metric", metric)
            metric = {**metric}
            jax.experimental.io_callback(callback, None, metric)

            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train

@hydra.main(version_base=None, config_path="config", config_name="ippo_ff_mpe_facmac")
def main(config):
    config = OmegaConf.to_container(config)
    
    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        name="prosocial-shareparam",
        tags=["IPPO", "FF"],
        config=config,
        mode=config["WANDB_MODE"],
    )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])   
    train_jit = jax.jit(make_train(config))
    out = jax.vmap(train_jit)(rngs)
    
    if config['SAVE_PATH'] is not None:
        env_name = config["ENV_NAME"]
        alg_name = "IPPO"
        def save_params(params: Dict, filename: Union[str, os.PathLike]) -> None:
            flattened_dict = flatten_dict(params, sep=',')
            save_file(flattened_dict, filename)

        model_state = out['runner_state'][0]
        params = jax.tree_map(lambda x: x[0], model_state.params) # save only params of the firt run
        save_dir = os.path.join(config['SAVE_PATH'], env_name, "prosocial")
        os.makedirs(save_dir, exist_ok=True)
        
        # path = ocp.test_utils.erase_and_create_empty('/tmp/my-checkpoints/')
        # checkpointer = ocp.StandardCheckpointer()
        # # 'checkpoint_name' must not already exist.
        # checkpointer.save(path / 'checkpoint_name', params)
        
        save_params(params, f'{save_dir}/{alg_name}.safetensors')
        print(f'Parameters of first batch saved in {save_dir}/{alg_name}.safetensors')
        
    print(out["metrics"].keys())
    print("updates shape", out["metrics"]["episode_returns"].shape)
    print(out["metrics"]["episode_returns"][:,-1,:,0]) # seed, steps, envs, agents
    print(out["metrics"]["episode_returns"][:,-1,:,0].mean(axis=1)) # seed, steps, envs, agents
    
    # show trajectory of returns
    # print("returns end", out["metrics"]["episode_returns"][0,-1,:,0])
    # print("returns middle", out["metrics"]["episode_returns"][0,4,:,0])
    # print("episode lengths", out["metrics"]["episode_lengths"][0,-1,:,0])
    # print("episode lengths beginning", out["metrics"]["episode_lengths"][0,0,:,0])
    
    #logging
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    updates_x = jnp.expand_dims(jnp.arange(out["metrics"]["episode_returns"].shape[1]),1)
    # returns_table = jnp.concatenate([updates_x, out["metrics"]["episode_returns"].mean(axis=(0,2))], axis=1)
    # returns_table = wandb.Table(data=returns_table.tolist(), columns=["updates"] + env.agents[:env.num_adversaries])
    for i in range(env.num_adversaries):
        returns_table = jnp.concatenate([updates_x, jnp.expand_dims(out["metrics"]["episode_returns"].mean(axis=(0,2))[:,i], 1)], axis=1)
        returns_table = wandb.Table(data=returns_table.tolist(), columns=["updates", "returns"+str(i)])
        wandb.log({
            # "returns_plot": wandb.plot.line_series(xs=jnp.arange(out["metrics"]["episode_returns"].shape[1]).tolist(), ys=out["metrics"]["episode_returns"].mean(axis=(0,2)).T.tolist(), keys = env.agents[:env.num_adversaries], title="Returns vs Updates", xname="Updates")
            "returns"+str(i)+"_plot": wandb.plot.line(returns_table, "updates", "returns"+str(i), title="returns"+str(i)+"_vs_updates"),
            # "returns": out["metrics"]["returned_episode_returns"][:,-1].mean(),
            # "total_loss_plot": wandb.plot.line(loss_table, "updates", "total_loss", title="total_loss_vs_updates"),
            # "actor_loss_plot": wandb.plot.line(loss_table, "updates", "actor_loss", title="actor_loss_vs_updates"),
            # "critic_loss_plot": wandb.plot.line(loss_table, "updates", "critic_loss", title="critic_loss_vs_updates"),
            # "entropy_plot": wandb.plot.line(loss_table, "updates", "entropy", title="entropy_vs_updates"),
        })
    
    returns_table = jnp.concatenate([updates_x, out["metrics"]["episode_returns"].mean(axis=(0,2))], axis=1)
    returns_table = wandb.Table(data=returns_table.tolist(), columns=["updates"] + env.agents[:env.num_adversaries])
    wandb.log({
        "returns_plot": wandb.plot.line_series(xs=jnp.arange(out["metrics"]["episode_returns"].shape[1]).tolist(), ys=out["metrics"]["episode_returns"].mean(axis=(0,2)).T.tolist(), keys = env.agents[:env.num_adversaries], title="Returns vs Updates", xname="Updates")
        # "returns"+str(i)+"_plot": wandb.plot.line(returns_table, "updates", "returns"+str(i), title="returns"+str(i)+"_vs_updates"),
        # "returns": out["metrics"]["returned_episode_returns"][:,-1].mean(),
        # "total_loss_plot": wandb.plot.line(loss_table, "updates", "total_loss", title="total_loss_vs_updates"),
        # "actor_loss_plot": wandb.plot.line(loss_table, "updates", "actor_loss", title="actor_loss_vs_updates"),
        # "critic_loss_plot": wandb.plot.line(loss_table, "updates", "critic_loss", title="critic_loss_vs_updates"),
        # "entropy_plot": wandb.plot.line(loss_table, "updates", "entropy", title="entropy_vs_updates"),
    })
    
    
if __name__ == "__main__":
    main()