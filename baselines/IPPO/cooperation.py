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
import orbax.checkpoint
from flax.training import orbax_utils
import chex
from flax.traverse_util import flatten_dict, unflatten_dict
from safetensors.flax import save_file, load_file


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
    return x


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def make_train(config, path0, path1):
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
            1
    )
    # env = FlattenObservationWrapper(env) # NOTE need a batchify wrapper
    env = LogWrapper(env, replace_info=True)

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    def train(rng):

        # INIT NETWORK
        # TODO doesn't work for non-homogenous agents
        # print(env.action_space(env.agents[0]).shape[0])
        network0 = ActorCritic(env.action_space(env.agents[0]).shape[0], activation=config["ACTIVATION"])
        network1 = ActorCritic(env.action_space(env.agents[0]).shape[0], activation=config["ACTIVATION"])
        rng, _rng0, _rng1 = jax.random.split(rng, num=3)
        # print("randoms", _rng0, _rng1)
        # print(env.observation_space(env.agents[0]).shape)
        # print(env.agents)
        # # init_x = jnp.zeros(env.observation_space(env.agents[0]).shape)
        # print("init x", init_x)
        
        def load_params(filename):
            flattened_dict = load_file(filename)
            return unflatten_dict(flattened_dict, sep=',')
        
        network_params0 = load_params(path0)
        network_params1 = load_params(path1)
        # network_params0 = network0.init(_rng0, init_x)
        # network_params1 = network1.init(_rng1, init_x)
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(config["LR"], eps=1e-5))

        train_state0 = TrainState.create(
            apply_fn=network0.apply,
            params=network_params0,
            tx=tx,
        )
        train_state1 = TrainState.create(
            apply_fn=network1.apply,
            params=network_params1,
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
                train_state0, train_state1, env_state, last_obs, rng = runner_state

                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                # print("obs_batch", obs_batch.shape)
                # print("obs_batch values", obs_batch)

                # SELECT ACTION
                rng, _rng0, _rng1 = jax.random.split(rng, num=3)

                pi0, value0 = network0.apply(train_state0.params, obs_batch[0])
                pi1, value1 = network1.apply(train_state1.params, obs_batch[1])
                # print("pi", pi)
                action0 = pi0.sample(seed=_rng0)
                action1 = pi1.sample(seed=_rng1)
                # print("action", action0, action1)
                log_prob0 = pi0.log_prob(action0)
                log_prob1 = pi1.log_prob(action1)
                action = [action0, action1]
                for _ in range(env.num_good_agents):
                    action.append(jnp.zeros(action0.shape))
                action = jnp.stack([a for a in action])
                env_act = unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents)
                # print("env_act", env_act)
                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step)(
                    rng_step, env_state, env_act,
                )
                # print("info before", info)
                # info = jax.tree_map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                # print("info", info)
                # print("SHAPE CHECK", batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()[0].shape, action0.shape, value0.shape, batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze()[0].shape, log_prob0.shape, obs_batch[0].shape, jax.tree_map(lambda x: x[:,0], info))
                transition0 = Transition(
                    batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()[0],
                    action0,
                    value0,
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze()[0],
                    log_prob0,
                    obs_batch[0],
                    jax.tree_map(lambda x: x[:,0], info),
                )
                transition1 = Transition(
                    batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()[1],
                    action1,
                    value1,
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze()[1],
                    log_prob1,
                    obs_batch[1],
                    jax.tree_map(lambda x: x[:,1], info),
                )
                runner_state = (train_state0, train_state1, env_state, obsv, rng)
                return runner_state, (transition0, transition1)

            runner_state, (traj_batch0, traj_batch1) = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )
            
            train_state0, train_state1, env_state, last_obs, rng = runner_state
            
            def callback(metric):
                wandb.log(
                    metric
                )

            update_state = (train_state0, train_state1, traj_batch0, traj_batch1, rng)

            train_state0 = update_state[0]
            train_state1 = update_state[1]
            metric0 = traj_batch0.info
            metric1 = traj_batch1.info
            rng = update_state[-1]

            # r0 = {"ratio0": loss_info0["ratio"][0,0].mean()}
            # jax.debug.print('ratio0 {x}', x=r0["ratio0"])
            # loss_info0 = jax.tree_map(lambda x: x.mean(), loss_info0)
            # loss_info1 = jax.tree_map(lambda x: x.mean(), loss_info1)
            # metric0 = jax.tree_map(lambda x: x.mean(), metric0)
            # metric1 = jax.tree_map(lambda x: x.mean(), metric1)
            metric = {"agent0":{**metric0}, "agent1":{**metric1}}
            jax.experimental.io_callback(callback, None, metric)
            runner_state = (train_state0, train_state1, env_state, last_obs, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state0, train_state1, env_state, obsv, _rng)
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
        name=config["NAME"],
        tags=["IPPO", "FF"],
        config=config,
        mode=config["WANDB_MODE"],
    )
    path0 = "ckpt/MPE_simple_facmac_v1/adversary0/IPPO.safetensors"
    path0BR = "ckpt/MPE_simple_facmac_v1/BRadversary0/IPPO.safetensors"
    path1 = "ckpt/MPE_simple_facmac_v1/adversary1/IPPO.safetensors"
    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_jit_actual = jax.jit(make_train(config, path0, path1))
    train_jit_br = jax.jit(make_train(config, path0BR, path1))
    out_actual = jax.vmap(train_jit_actual)(rngs)
    out_br = jax.vmap(train_jit_br)(rngs)
    print("returned episode returns", out_actual["metrics"]["agent0"]["returned_discounted_episode_returns"].shape)
    # print("trajectory return", out_actual["metrics"]["agent0"]["returned_episode_returns"][0,:,:,0])
    # print("trajectory lengths", out_actual["metrics"]["agent0"]["returned_episode_lengths"][0,:,:,0])
    # print()
    # print("returned episode returns", out_br["metrics"]["agent0"]["returned_episode_returns"].shape)
    # print("trajectory return", out_br["metrics"]["agent0"]["returned_episode_returns"][0,:,:,0])
    # print("trajectory lengths", out_br["metrics"]["agent0"]["returned_episode_lengths"][0,:,:,0])
    
    print("Actual Welfare", out_actual["metrics"]["agent0"]["returned_episode_returns"][:,:,-1,:].mean())
    print("BR Welfare", out_br["metrics"]["agent0"]["returned_episode_returns"][:,:,-1,:].mean())
    
    print("Actual Welfare (Discounted)", out_actual["metrics"]["agent0"]["returned_discounted_episode_returns"][:,:,-1,:].mean())
    print("BR Welfare (Discounted)", out_br["metrics"]["agent0"]["returned_discounted_episode_returns"][:,:,-1,:].mean())
    return
    # print(out)
    # save params
    # env_name = config["ENV_NAME"]
    # alg_name = "IPPO"
    # if config['SAVE_PATH'] is not None:

    #     def save_params(params: Dict, filename: Union[str, os.PathLike]) -> None:
    #         flattened_dict = flatten_dict(params, sep=',')
    #         save_file(flattened_dict, filename)

    #     model_state = out['runner_state'][0]
    #     params = jax.tree_map(lambda x: x[0], model_state.params) # save only params of the first run
    #     save_dir = os.path.join(config['SAVE_PATH'], env_name, "adversary0")
    #     os.makedirs(save_dir, exist_ok=True)
    #     save_params(params, f'{save_dir}/{alg_name}.safetensors')
        
    #     model_state = out['runner_state'][1]
    #     params = jax.tree_map(lambda x: x[0], model_state.params) # save only params of the first run
    #     save_dir = os.path.join(config['SAVE_PATH'], env_name, "adversary1")
    #     os.makedirs(save_dir, exist_ok=True)
    #     save_params(params, f'{save_dir}/{alg_name}.safetensors')
    #     print(f'Parameters of first batch saved in {save_dir}/{alg_name}.safetensors')
        
        # train_state = out['runner_state'][0]
        # save_args = orbax_utils.save_args_from_target(train_state)
        # orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        # save_loc = 'tmp/orbax/checkpoint_rnn'
        # orbax_checkpointer.save(save_loc, train_state, save_args=save_args)

    # logging
    updates_x0 = jnp.arange(out["metrics"]["agent0"]["total_loss"][0].shape[0])
    updates_x1 = jnp.arange(out["metrics"]["agent1"]["total_loss"][0].shape[0])
    loss_table0 = jnp.stack([updates_x0, out["metrics"]["agent0"]["total_loss"].mean(axis=0), out["metrics"]["agent0"]["actor_loss"].mean(axis=0), out["metrics"]["agent0"]["critic_loss"].mean(axis=0), out["metrics"]["agent0"]["entropy"].mean(axis=0), out["metrics"]["agent0"]["ratio"].mean(axis=0)], axis=1)
    loss_table1 = jnp.stack([updates_x1, out["metrics"]["agent1"]["total_loss"].mean(axis=0), out["metrics"]["agent1"]["actor_loss"].mean(axis=0), out["metrics"]["agent1"]["critic_loss"].mean(axis=0), out["metrics"]["agent1"]["entropy"].mean(axis=0), out["metrics"]["agent1"]["ratio"].mean(axis=0)], axis=1)        
    loss_table0 = wandb.Table(data=loss_table0.tolist(), columns=["updates", "total_loss", "actor_loss", "critic_loss", "entropy", "ratio"])
    loss_table1 = wandb.Table(data=loss_table1.tolist(), columns=["updates", "total_loss", "actor_loss", "critic_loss", "entropy", "ratio"])
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
    
    # import pdb;

    # pdb.set_trace()


if __name__ == "__main__":
    main()