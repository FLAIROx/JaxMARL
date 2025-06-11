""" 
Based on PureJaxRL Implementation of PPO
"""

import wandb
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
import time
import sys
import jaxmarl.environments.mabrax.mabrax_env

from jax2onnx import to_onnx, onnx_function

class EarlyTermination(Exception): 
    pass

class ActorModule(nn.Module):
    action_dim: int
    activation: str = "tanh"
    actor_arch: Sequence[int] = None


    @nn.compact
    def __call__(self, x):
        act_fn = nn.relu if self.activation == "relu" else nn.tanh
        a = x
        for h in self.actor_arch or [128, 64, 64]:
            a = nn.Dense(h, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(a)
            a = act_fn(a)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(a)
        return actor_mean

class CriticModule(nn.Module):
    activation: str = "tanh"
    critic_arch: Sequence[int] = None


    @nn.compact
    def __call__(self, x):
        act_fn = nn.relu if self.activation == "relu" else nn.tanh
        c = x
        for h in self.critic_arch or [128, 128, 128, 128]:
            c = nn.Dense(h, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(c)
            c = act_fn(c)
        c = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(c)
        return jnp.squeeze(c, axis=-1)

class ActorCritic(nn.Module):
    action_dim: int
    activation: str = "tanh"
    actor_arch: Sequence[int] = None
    critic_arch: Sequence[int] = None

    def setup(self):
        self.actor_module = ActorModule(action_dim=self.action_dim,
                                        activation=self.activation,
                                        actor_arch=self.actor_arch)
        self.critic_module = CriticModule(activation=self.activation,
                                          critic_arch=self.critic_arch)
        self.log_std = self.param('log_std', nn.initializers.zeros, (self.action_dim,))

    def __call__(self, x):
        actor_mean = self.actor_module(x)
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(self.log_std))
        critic = self.critic_module(x)
        return pi, critic
   
    def actor_forward(self, x):
        # Returns actor output only
        return self.actor_module(x)
   
    def critic_forward(self, x):
        # Returns critic value only
        return self.critic_module(x)

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
    #print('max_dim', max_dim)
    def pad(z):
        return jnp.concatenate([z, jnp.zeros(z.shape[:-1] + (max_dim - z.shape[-1],))], -1)

    x = jnp.stack([x[a] if x[a].shape[-1] == max_dim else pad(x[a]) for a in agent_list])
    return x.reshape((num_actors, -1))

def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}

def make_train(config, rng_init):
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    
    env = LogWrapper(env, replace_info=True)
    
    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    # INIT NETWORK 
    network = ActorCritic(
        action_dim=env.action_space(env.agents[0]).shape[0],
        activation=config["ACTIVATION"],
        actor_arch=config.get("ACTOR_ARCH", [128, 64, 64]),
        critic_arch=config.get("CRITIC_ARCH", [128, 128, 128])
    )
    print('Network initialized with architectures:', network.actor_arch, network.critic_arch)

    max_dim = jnp.argmax(jnp.array([env.observation_space(a).shape[-1] for a in env.agents]))
    init_x = jnp.zeros(env.observation_space(env.agents[max_dim]).shape)
    network_params = network.init(rng_init, init_x)
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

    def train(rng):        
        
        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset)(reset_rng)


        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, update_count, rng = runner_state

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

                info = jax.tree.map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                transition = Transition(
                    batchify(done, env.agents, config["NUM_ACTORS"]).squeeze(),
                    action,
                    value,
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob,
                    obs_batch,
                    info,
                )
                runner_state = (train_state, env_state, obsv, update_count, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )
            print('traj_batch', traj_batch)
            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, update_count, rng = runner_state
            last_obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
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
                        return total_loss, (value_loss, loss_actor, entropy, ratio)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    
                    loss_info = {
                        "total_loss": total_loss[0],
                        "actor_loss": total_loss[1][1],
                        "critic_loss": total_loss[1][0],
                        "entropy": total_loss[1][2],
                        "ratio": total_loss[1][3],
                    }
                    
                    return train_state, loss_info

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ACTORS"]
                ), "batch size must be equal to number of steps * number of actors"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree.map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree.map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, loss_info = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, loss_info

            def callback(metric):
                wandb.log(metric, step=metric["update_step"])
                global last_interval_log_time
                global last_termination_threshhold
                current_time = time.time()
                if current_time - last_interval_log_time >= 60:
                    r_lengths = metric.get("returned_episode_lengths", None)
                    if r_lengths is not None:
                        # Check if r_lengths is 0-dimensional and extract its value accordingly.
                        if hasattr(r_lengths, "ndim") and r_lengths.ndim == 0:
                            interval_value = r_lengths.item()
                        else:
                            interval_value = r_lengths[-1]
                        wandb.log({"episode_length_interval": interval_value, "termination_threshold": last_termination_threshhold}, step=metric["update_step"])
                    
                        # Trigger early termination via a custom exception
                        # if interval_value < last_termination_threshhold:
                        #     wandb.log({"early_termination": True}, step=metric["update_step"])
                        #     print("Early termination triggered.")
                        #     wandb.finish(exit_code=0)
                        #     raise EarlyTermination("Terminating training.")
                        # # Update the termination threshold for the next interval
                        # delta = jnp.clip(interval_value - last_termination_threshhold, 10, 20)
                        # last_termination_threshhold += delta * 0.2 * (1.1 - np.clip(interval_value/600,0,1))
                    
                    last_interval_log_time = current_time
                    

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]

            update_count = update_count + 1
            r0 = {"ratio0": loss_info["ratio"][0,0].mean()}
            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)
            metric = jax.tree.map(lambda x: x.mean(), metric)
            metric["update_step"] = update_count
            metric["env_step"] = update_count * config["NUM_STEPS"] * config["NUM_ENVS"]
            metric = {**metric, **loss_info, **r0}
            jax.experimental.io_callback(callback, None, metric)
            runner_state = (train_state, env_state, last_obs, update_count, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, 0, _rng)
        try:
            runner_state, metric = jax.lax.scan(
                _update_step, runner_state, None, config["NUM_UPDATES"]
            )
        except EarlyTermination:
            # Early termination: return current state instead of raising an error.
            return {"runner_state": runner_state, "metrics": {}}
        return {"runner_state": runner_state, "metrics": metric}

    return train

last_interval_log_time = 0
last_termination_threshhold = 0

@hydra.main(version_base=None, config_path="config", config_name="ippo_ff_mabrax")
def main(config):

    config = OmegaConf.to_container(config)

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["IPPO", "FF"],
        config=config,
        mode=config["WANDB_MODE"],
    )

    rng = jax.random.PRNGKey(config["SEED"])
    with jax.disable_jit(config["DISABLE_JIT"]):
        rng, _rng = jax.random.split(rng)
        train_jit = jax.jit(make_train(config, _rng),  device=jax.devices()[config["DEVICE"]])
        out = train_jit(rng)
    
    
    # mean_returns = out["metrics"]["returned_episode_returns"].mean(-1).reshape(-1)
    # x = np.arange(len(mean_returns)) * config["NUM_ACTORS"]
    # plt.plot(x, mean_returns)
    # plt.xlabel("Timestep")
    # plt.ylabel("Return")
    # plt.savefig(f'mabrax_ippo_ret.png')
    
    # import pdb; pdb.set_trace()

if __name__ == "__main__":
    main()