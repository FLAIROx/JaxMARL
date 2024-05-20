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
# import orbax.checkpoint
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

class MultiFacmacMPE(SimpleFacmacMPE):
    """Log the episode returns and lengths.
    NOTE for now for envs where agents terminate at the same time.
    """

    def __init__(
        self,
        num_good_agents=1,
        num_adversaries=3,
        num_landmarks=2,
        view_radius=1.5,  # set -1 to deactivate
        score_function="sum"
    ):
        super().__init__( 
        num_good_agents,
        num_adversaries,
        num_landmarks,
        view_radius, 
        score_function)
        
    def rewards(self, state: State) -> Dict[str, float]:
        @partial(jax.vmap, in_axes=(0, None))
        def _collisions(agent_idx: int, other_idx: int):
            return jax.vmap(self.is_collision, in_axes=(None, 0, None))(
                agent_idx,
                other_idx,
                state,
            )

        c = _collisions(
            jnp.arange(self.num_good_agents) + self.num_adversaries,
            jnp.arange(self.num_adversaries),
        )  # [agent, adversary, collison]

        def _good(aidx: int, collisions: chex.Array):
            rew = -10 * jnp.sum(collisions[aidx])

            mr = jnp.sum(self.map_bounds_reward(jnp.abs(state.p_pos[aidx])))
            rew -= mr
            return rew

        # ad_rew = 10 * jnp.sum(c)
        
        def _adv(aidx: int, collisions: chex.Array):
            rew = 10 * jnp.sum(collisions[:,aidx])
            
            return rew

        rew = {a: _adv(i, c)
                for i, a in enumerate(self.adversaries)}
        
        rew.update(
            {
                a: _good(i + self.num_adversaries, c)
                for i, a in enumerate(self.good_agents)
            }
        )
        # print("rewards!", rew)
        return rew
    
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
    env = MultiFacmacMPE(**config["ENV_KWARGS"])
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
                print("last_obs", last_obs)
                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                print("obs_batch", obs_batch.shape)
                # SELECT ACTION
                rng, _rng = jax.random.split(rng)

                pi, value = network.apply(train_state.params, obs_batch)
                action = pi.sample(seed=_rng)
                # print("action and pi", action.shape, pi.shape)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents)
                print("env_act", env_act)
                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step)(
                    rng_step, env_state, env_act,
                )
                print("SHAPE CHECK", batchify(done, env.agents, config["NUM_ACTORS"]).squeeze().shape, action.shape, value.shape, batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze().shape, log_prob.shape, obs_batch.shape, info)
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
                    print("SHAPE CHECK gae:", done.shape, value.shape, reward.shape, next_value.shape)
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

            print("traj batch last val", traj_batch, last_val.shape)
            advantages, targets = _calculate_gae(traj_batch, last_val)
            print("advantages", advantages.shape)
            
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
                    
                    loss_info = {
                        "total_loss": total_loss[0],
                        "actor_loss": total_loss[1][1],
                        "critic_loss": total_loss[1][0],
                        "entropy": total_loss[1][2],
                    }
                    
                    return train_state, loss_info

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                        batch_size == config["NUM_STEPS"] * config["NUM_ADVERSARIES"]
                ), "batch size must be equal to number of steps * number of actors"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                # print("traj_batch shape", traj_batch)
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
                train_state, loss_info = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, loss_info

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
            # print("metric before", metric)
            metric = jax.tree_map(lambda x: x.reshape((config["NUM_STEPS"], config["NUM_ENVS"], -1)), metric)
            # print("metric after", metric)
            metrics = {}
            for i in range(env.num_adversaries):
                m = jax.tree_map(lambda x: x[:,:,i].mean(), metric)
                agentname = "agent" + str(i)
                metrics.update({agentname:{**m, **loss_info}})
                # print(metrics)
            jax.experimental.io_callback(callback, None, metrics)
            
            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metrics

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
        name="selfish-paramsharing",
        tags=["IPPO", "FF"],
        config=config,
        mode=config["WANDB_MODE"],
    )
    
    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])    
    train_jit = jax.jit(make_train(config))
    out = jax.vmap(train_jit)(rngs)
    # print(out)
    # save params
    env_name = config["ENV_NAME"]
    alg_name = "IPPO"
    if config['SAVE_PATH'] is not None:

        def save_params(params: Dict, filename: Union[str, os.PathLike]) -> None:
            flattened_dict = flatten_dict(params, sep=',')
            save_file(flattened_dict, filename)
            
        

        model_state = out['runner_state'][0]
        params = jax.tree_map(lambda x: x[0], model_state.params) # save only params of the firt run
        save_dir = os.path.join(config['SAVE_PATH'], env_name, "selfish")
        os.makedirs(save_dir, exist_ok=True)
        
        # path = ocp.test_utils.erase_and_create_empty('/tmp/my-checkpoints/')
        # checkpointer = ocp.StandardCheckpointer()
        # # 'checkpoint_name' must not already exist.
        # checkpointer.save(path / 'checkpoint_name', params)
        
        save_params(params, f'{save_dir}/{alg_name}.safetensors')
        print(f'Parameters of first batch saved in {save_dir}/{alg_name}.safetensors')

    # logging
    print("OUT METRICS", out["metrics"]["agent0"]["returned_episode_returns"].shape, out["metrics"]["agent0"]["returned_episode_returns"][0].shape[0])
    # updates_x0 = jnp.arange(out["metrics"]["agent0"]["total_loss"][0].shape[0])
    # updates_x1 = jnp.arange(out["metrics"]["agent1"]["total_loss"][0].shape[0])
    # loss_table0 = jnp.stack([updates_x0, out["metrics"]["agent0"]["total_loss"].mean(axis=0), out["metrics"]["agent0"]["actor_loss"].mean(axis=0), out["metrics"]["agent0"]["critic_loss"].mean(axis=0), out["metrics"]["agent0"]["entropy"].mean(axis=0), out["metrics"]["agent0"]["ratio"].mean(axis=0)], axis=1)
    # loss_table1 = jnp.stack([updates_x1, out["metrics"]["agent1"]["total_loss"].mean(axis=0), out["metrics"]["agent1"]["actor_loss"].mean(axis=0), out["metrics"]["agent1"]["critic_loss"].mean(axis=0), out["metrics"]["agent1"]["entropy"].mean(axis=0), out["metrics"]["agent1"]["ratio"].mean(axis=0)], axis=1)        
    # loss_table0 = wandb.Table(data=loss_table0.tolist(), columns=["updates", "total_loss", "actor_loss", "critic_loss", "entropy", "ratio"])
    # loss_table1 = wandb.Table(data=loss_table1.tolist(), columns=["updates", "total_loss", "actor_loss", "critic_loss", "entropy", "ratio"])
    updates_x0 = jnp.arange(out["metrics"]["agent0"]["returned_episode_returns"][0].shape[0])
    updates_x1 = jnp.arange(out["metrics"]["agent1"]["returned_episode_returns"][0].shape[0])
    returns_table0 = jnp.stack([updates_x0, out["metrics"]["agent0"]["returned_episode_returns"].mean(axis=0)], axis=1)
    returns_table1 = jnp.stack([updates_x1, out["metrics"]["agent1"]["returned_episode_returns"].mean(axis=0)], axis=1)
    returns_table0 = wandb.Table(data=returns_table0.tolist(), columns=["updates0", "returns0"])
    returns_table1 = wandb.Table(data=returns_table1.tolist(), columns=["updates1", "returns1"])
    wandb.log({
        "returns_plot0": wandb.plot.line(returns_table0, "updates0", "returns0", title="returns_vs_updates0"),
        # "returns0": out["metrics"]["returned_episode_returns"][:,-1].mean(),
        # "total_loss_plot0": wandb.plot.line(loss_table0, "updates", "total_loss", title="total_loss_vs_updates0"),
        # "actor_loss_plot0": wandb.plot.line(loss_table0, "updates", "actor_loss", title="actor_loss_vs_updates0"),
        # "critic_loss_plot0": wandb.plot.line(loss_table0, "updates", "critic_loss", title="critic_loss_vs_updates0"),
        # "entropy_plot0": wandb.plot.line(loss_table0, "updates", "entropy", title="entropy_vs_updates0"),
        # "ratio_plot0": wandb.plot.line(loss_table0, "updates", "ratio", title="ratio_vs_updates0"),
        "returns_plot1": wandb.plot.line(returns_table1, "updates1", "returns1", title="returns_vs_updates1"),
        # "returns1": out["metrics"]["returned_episode_returns"][:,-1].mean(),
        # "total_loss_plot1": wandb.plot.line(loss_table1, "updates", "total_loss", title="total_loss_vs_updates1"),
        # "actor_loss_plot1": wandb.plot.line(loss_table1, "updates", "actor_loss", title="actor_loss_vs_updates1"),
        # "critic_loss_plot1": wandb.plot.line(loss_table1, "updates", "critic_loss", title="critic_loss_vs_updates1"),
        # "entropy_plot1": wandb.plot.line(loss_table1, "updates", "entropy", title="entropy_vs_updates1"),
        # "ratio_plot1": wandb.plot.line(loss_table1, "updates", "ratio", title="ratio_vs_updates1"),
    })
    
    
    
if __name__ == "__main__":
    main()