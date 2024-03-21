import jax 
import jax.numpy as jnp
from typing import Optional, NamedTuple, Sequence, Dict, Tuple, Union
import chex
from optax import OptState
from flax.core.frozen_dict import FrozenDict
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import distrax
import optax
from optax import OptState
import functools

from jaxmarl.environments.multi_agent_env import State
from jaxmarl.environments.smax import map_name_to_scenario, HeuristicEnemySMAX
from jaxmarl.wrappers.baselines import SMAXLogWrapper, AddAgentID


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
        rnn_state = jnp.where(
            resets[:, jnp.newaxis],
            self.initialize_carry(*rnn_state.shape),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))

class ActorRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones, avail_actions = x
        embedding = nn.Dense(
            128, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(self.config["GRU_HIDDEN_DIM"], kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        masked_logits = jnp.where(
            avail_actions,
            actor_mean,
            jnp.finfo(jnp.float32).min,
        )
        # unavail_actions = 1 - avail_actions
        # action_logits = actor_mean - (unavail_actions * 1e10)
        # TODO same as mava (discrete action head)

        pi = distrax.Categorical(logits=masked_logits)
        return hidden, pi

class CriticRNN(nn.Module):
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        embedding = nn.Dense(
            128, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        critic = nn.Dense(128, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
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
    info: jnp.ndarray
    avail_actions: jnp.ndarray
    terminated: jnp.ndarray

class Params(NamedTuple):
    """Parameters of an actor critic network."""
    actor_params: FrozenDict
    critic_params: FrozenDict

class OptStates(NamedTuple):
    """OptStates of actor critic learner."""
    actor_opt_state: OptState
    critic_opt_state: OptState


class HiddenStates(NamedTuple):
    """Hidden states for an actor critic learner."""
    actor_hstate: chex.Array
    critic_hstate: chex.Array

class RNNRunnerState(NamedTuple):
    """State of the `Learner` for recurrent architectures."""
    update_step: int
    params: Params
    opt_states: OptStates
    env_state: State
    obs: jnp.ndarray
    done: jnp.ndarray
    episode_done: jnp.ndarray
    hstates: HiddenStates
    key: chex.PRNGKey

def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}

class SMAXCoinFitness(object):
    def __init__(
        self,
        config: Dict,
        num_rollouts: int = 16,
        test: bool = False,
        n_devices: Optional[int] = None,
    ):
        self.config = config
        self.num_rollouts = num_rollouts
        self.test = test            
        
        self.scenario = map_name_to_scenario(self.config["MAP_NAME"])
        self.env = HeuristicEnemySMAX(scenario=self.scenario, **self.config["ENV_KWARGS"]) 
        
        self.config["NUM_ACTORS"] = self.config["NUM_ENVS"] * self.env.num_agents
        self.config["NUM_UPDATES"] = (
            config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
        )
        self.config["MINIBATCH_SIZE"] = (
            config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
        )
        
        self.env = SMAXLogWrapper(self.env)
        if config["ADD_AGENT_ID"]:
            self.env = AddAgentID(self.env)
        
        self.steps_per_member = self.env.max_steps * num_rollouts  
        
        self.action_shape = self.env.num_ally_actions
        self.input_shape = self.env.observation_space(self.env.agents[0]).shape
        print('input_shape:', self.input_shape, 'action_shape:', self.action_shape) 
        
        self.actor_network = ActorRNN(self.env.action_space(self.env.agents[0]).n, config=self.config)
        self.critic_network = CriticRNN(config=self.config)
        
        if n_devices is None:
            self.n_devices = jax.local_device_count()
        else:
            self.n_devices = n_devices
            
        self.total_env_steps = 0
    
    def set_apply_fn(self, network_apply, carry_init=None):
        """ Set the network forward function """
        self.network = network_apply 
        
        if carry_init is not None:
            raise NotImplementedError
            self.single_rollout = self.rollout_rnn
            self.carry_init = carry_init
        else:
            self.single_rollout = self.rollout_ffw
        self.rollout_repeats = jax.vmap(self.single_rollout, in_axes=(0, None))
        self.rollout_pop = jax.vmap(self.rollout_repeats, in_axes=(None, 0))
        # pmap over popmembers if > 1 device is available - otherwise pmap
        if self.n_devices > 1:
            self.rollout_map = self.rollout_pmap
            print(
                f"GymFitness: {self.n_devices} devices detected. Please make"
                " sure that the ES population size divides evenly across the"
                " number of devices to pmap/parallelize over."
            )
        else:
            self.rollout_map = self.rollout_pop
            
    def rollout_pmap(
        self, rng_input: chex.PRNGKey, reward_network_params: chex.ArrayTree
    ):
        """Parallelize rollout across devices. Split keys/reshape correctly."""
        keys_pmap = jnp.tile(rng_input, (self.n_devices, 1, 1))
        rew_dev, steps_dev = jax.pmap(self.rollout_pop)(
            keys_pmap, reward_network_params
        )
        rew_re = rew_dev.reshape(-1, self.num_rollouts)
        steps_re = steps_dev.reshape(-1, self.num_rollouts)
        return rew_re, steps_re
        
    def rollout(self, rng_input: chex.PRNGKey, reward_network_params: chex.ArrayTree):
        """Placeholder fn call for rolling out a population for multi-evals."""
        rng_pop = jax.random.split(rng_input, self.num_rollouts)
        scores = jax.jit(self.rollout_map)(rng_pop, reward_network_params)
        # Update total step counter using only transitions before termination
        # self.total_env_steps += masks.sum()
        return scores
    
    def rollout_ffw(
        self, rng_input: chex.PRNGKey, reward_network_params: chex.ArrayTree
    ):
        """ Train an agent """
        
        def linear_schedule(count):
            frac = (
                1.0
                - (count // (self.config["NUM_MINIBATCHES"] * self.config["UPDATE_EPOCHS"]))
                / self.config["NUM_UPDATES"]
            )
            return self.config["LR"] * frac
        
        # INIT PARAMS
        rng, _rng = jax.random.split(rng_input)
        init_x_actor = (
            jnp.zeros(
                (1, self.config["NUM_ENVS"], self.env.observation_space(self.env.agents[0]).shape[0])
            ),
            jnp.zeros((1, self.config["NUM_ENVS"])),
            jnp.zeros((1, self.config["NUM_ENVS"], self.env.action_space(self.env.agents[0]).n)),
        )
        init_hstate = ScannedRNN.initialize_carry(self.config["NUM_ENVS"], self.config["GRU_HIDDEN_DIM"])
        actor_params = self.actor_network.init(_rng, init_hstate, init_x_actor)
        
        rng, _rng = jax.random.split(rng)
        critic_params = self.critic_network.init(_rng, init_hstate, (init_x_actor[0], init_x_actor[1]))
        
        if self.config["ANNEAL_LR"]:
            actor_optim = optax.chain(
                optax.clip_by_global_norm(self.config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
            critic_optim = optax.chain(
                optax.clip_by_global_norm(self.config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            actor_optim = optax.chain(
                optax.clip_by_global_norm(self.config["MAX_GRAD_NORM"]),
                optax.adam(self.config["LR"], eps=1e-5),
            )
            critic_optim = optax.chain(
                optax.clip_by_global_norm(self.config["MAX_GRAD_NORM"]),
                optax.adam(self.config["LR"], eps=1e-5),
            )
            
        params=Params(
            actor_params=actor_params,
            critic_params=critic_params
        )
        opt_states=OptStates(
            actor_opt_state=actor_optim.init(actor_params),
            critic_opt_state=critic_optim.init(critic_params),
        )
        
        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, self.config["NUM_ENVS"])
        obsv, env_state = jax.vmap(self.env.reset, in_axes=(0,))(reset_rng)
        actor_init_hstate = ScannedRNN.initialize_carry(self.config["NUM_ACTORS"], self.config["GRU_HIDDEN_DIM"])
        critic_init_hstate = ScannedRNN.initialize_carry(self.config["NUM_ACTORS"], self.config["GRU_HIDDEN_DIM"])
        init_hstates = HiddenStates(
            actor_hstate=actor_init_hstate,
            critic_hstate=critic_init_hstate,
        )
        
        # TRAIN LOOP
        def _update_step(runner_state: RNNRunnerState, unused):
            
            # COLLECT TRAJECTORIES
            def _env_step(runner_state: RNNRunnerState, unused):
                update_step, params, opt_states, env_state, last_obs, last_done, last_episode_done, hstates, rng = runner_state

                # RUN NETWORKS
                rng, policy_rng = jax.random.split(rng)
                avail_actions = jax.vmap(self.env.get_avail_actions)(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(
                    batchify(avail_actions, self.env.agents, self.config["NUM_ACTORS"])
                )
                obs_batch = batchify(last_obs, self.env.agents, self.config["NUM_ACTORS"])
                ac_in = (
                    obs_batch[jnp.newaxis, :],
                    last_done[jnp.newaxis, :],
                    avail_actions,
                )
                actor_hstate, pi = self.actor_network.apply(params.actor_params, hstates.actor_hstate, ac_in)
                critic_hstate, value = self.critic_network.apply(params.critic_params, hstates.critic_hstate, (ac_in[0], ac_in[1]))
                
                action = pi.sample(seed=policy_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(
                    action, self.env.agents, self.config["NUM_ENVS"], self.env.num_agents
                )
                env_act = {k: v.squeeze() for k, v in env_act.items()}

                # STEP ENV
                rng, rng_step = jax.random.split(rng)
                rng_step = jax.random.split(rng_step, self.config["NUM_ENVS"])
                obs, env_state, reward, done, info = jax.vmap(
                    self.env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)
                info = jax.tree_map(lambda x: x.reshape((self.config["NUM_ACTORS"])), info)
                episode_done = done["__all__"]
                done_batch = batchify(done, self.env.agents, self.config["NUM_ACTORS"]).squeeze()
                reward_batch = batchify(reward, self.env.agents, self.config["NUM_ACTORS"]).squeeze()
                
                rn_in = jnp.concatenate((obs_batch, reward_batch[:, None]), axis=1)
                print('rn_in shape', rn_in.shape)
                learned_reward = self.network(reward_network_params, rn_in[None])
                print('learned reward shape', learned_reward.shape)
                ## term - mask out timesteps when the agent didn't act
                # if agent was done in the last timestep and the episode is not over then this action is invalid
                # but if the episode ends, then this action is valid 
                
                last_done = last_done.reshape((self.config["NUM_ENVS"], -1))
                last_ep_done = jnp.all(last_done, axis=1)
                term = last_done & ~last_ep_done[:, None]
                term = term.reshape((-1,))
                
                transition = Transition(
                    jnp.tile(done["__all__"], self.env.num_agents),
                    done_batch,
                    action.squeeze(),
                    value.squeeze(),
                    learned_reward,
                    log_prob.squeeze(),
                    obs_batch,
                    info,
                    avail_actions,
                    term
                )
                
                hstates = HiddenStates(actor_hstate, critic_hstate)
                runner_state = RNNRunnerState(
                    update_step, params, opt_states, env_state, obs, done_batch, episode_done, hstates, rng
                )                
                return runner_state, transition

            initial_hstates = runner_state.hstates
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, self.config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            update_step, params, opt_states, env_state, last_obs, last_done, last_episode_done, hstates, rng = runner_state
            last_obs_batch = batchify(last_obs, self.env.agents, self.config["NUM_ACTORS"])
            ac_in = (
                last_obs_batch[jnp.newaxis, :],
                last_done[jnp.newaxis, :],
            )
            _, last_val = self.critic_network.apply(params.critic_params, hstates.critic_hstate, ac_in)
            last_val = last_val.squeeze()  # mava here masks out the terminal states but surely unnecessary?
            last_val = jnp.where(last_done, jnp.zeros_like(last_val), last_val)
            # TODO mava's masking
            
            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.global_done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + self.config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + self.config["GAMMA"] * self.config["GAE_LAMBDA"] * (1 - done) * gae
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
            def _update_epoch(update_state: Tuple, unused):
                
                def _update_minbatch(train_state: Tuple, batch_info: Tuple):
                    
                    params, opt_states = train_state
                    init_hstates, traj_batch, advantages, targets = batch_info

                    def _actor_loss_fn(
                        actor_params: FrozenDict,
                        actor_opt_state: OptState,
                        actor_init_hstate: chex.Array,
                        traj_batch,
                        gae: chex.Array
                    ):
                        # RERUN NETWORK
                        _, pi = self.actor_network.apply(
                            actor_params,
                            actor_init_hstate.squeeze(),
                            (traj_batch.obs, traj_batch.done, traj_batch.avail_actions),
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - self.config["CLIP_EPS"],
                                1.0 + self.config["CLIP_EPS"],
                            )
                            * gae
                        )
                        # TODO add back in done masking on the mean 
                        print('loss actor shape', loss_actor1.shape, 'term shape', traj_batch.terminated.shape)
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean(where=(1-traj_batch.done))
                        entropy = pi.entropy().mean(where=(1-traj_batch.done))

                        total_loss = loss_actor - self.config["ENT_COEF"] * entropy
                        return total_loss, (loss_actor, entropy, ratio)
                    
                    def _critic_loss_fn(critic_params: FrozenDict,
                                        critic_opt_state: OptState,
                                        critic_init_hstate: chex.Array,
                                        traj_batch,
                                        targets: chex.Array):
                        # RERUN NETWORK
                        _, value = self.critic_network.apply(
                            critic_params,
                            critic_init_hstate.squeeze(),
                            (traj_batch.obs, traj_batch.done),
                        )
                        
                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-self.config["CLIP_EPS"], self.config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(
                            value_losses, value_losses_clipped
                        ).mean(where=(1-traj_batch.done)) 
                        
                        total_loss = self.config["VF_COEF"] * value_loss
                        return total_loss, (value_loss)

                    actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                    actor_loss_info, actor_grads = actor_grad_fn(
                        params.actor_params, opt_states.actor_opt_state, init_hstates.actor_hstate, traj_batch, advantages
                    )
                    
                    critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                    critic_loss_info, critic_grads = critic_grad_fn(
                        params.critic_params, opt_states.critic_opt_state, init_hstates.critic_hstate, traj_batch, targets
                    )

                    actor_updates, actor_new_opt_state = actor_optim.update(
                        actor_grads, opt_states.actor_opt_state
                    )
                    actor_new_params = optax.apply_updates(params.actor_params, actor_updates)
                    
                    critic_updates, critic_new_opt_state = critic_optim.update(
                        critic_grads, opt_states.critic_opt_state
                    )
                    critic_new_params = optax.apply_updates(params.critic_params, critic_updates)
                    
                    new_params = Params(actor_params=actor_new_params, critic_params=critic_new_params)
                    new_opt_state = OptStates(actor_opt_state=actor_new_opt_state, critic_opt_state=critic_new_opt_state)
                    
                    total_loss = actor_loss_info[0] + critic_loss_info[0]
                    loss_info = {
                        "total_loss": total_loss,
                        "value_loss": critic_loss_info[1],
                        "actor_loss": actor_loss_info[1][0],
                        "entropy": actor_loss_info[1][1],
                        "ratio": actor_loss_info[1][2],
                    }
                    
                    return (new_params, new_opt_state), loss_info

                (
                    params,
                    opt_states,
                    init_hstates,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state
                rng, shuffle_rng = jax.random.split(rng)

                # adding an additional "fake" dimensionality to perform minibatching correctly
                # NOTE: mava does this differently, reshaping into "recurrent chunk size" and then permuting
                # TODO: look at the shapes of mava and jaxmarl
                # NOTE: mava also handles hstates differently, use the one stored in traj_batch
                init_hstates = jax.tree_map(
                    lambda x: jnp.reshape(
                        x, (1, self.config["NUM_ACTORS"], -1)
                    ),
                    init_hstates,
                )                
                batch = (
                    init_hstates,
                    traj_batch,
                    advantages.squeeze(),
                    targets.squeeze(),
                )
                permutation = jax.random.permutation(shuffle_rng, self.config["NUM_ACTORS"])
                print('batch advantage shape', batch[2].shape)
                # advantage shape is (num_steps, num_actors)
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )

                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], self.config["NUM_MINIBATCHES"], -1]
                            + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )
                print('minibatch advantage shape', minibatches[2].shape)
                # advantage shape is (num_minibatches, num_steps, -1)
                
                (params, opt_states), loss_info = jax.lax.scan(
                    _update_minbatch, (params, opt_states), minibatches
                )
                init_hstates = jax.tree_map(
                    lambda x: x.squeeze(), init_hstates
                )
                update_state = (
                    params,
                    opt_states,
                    init_hstates,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, loss_info

            update_state = (
                params,
                opt_states,
                initial_hstates,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, self.config["UPDATE_EPOCHS"]
            )
            
            params, opt_states, _, _, _, _, rng = update_state
            
            metric = traj_batch.info
            metric = jax.tree_map(
                lambda x: x.reshape(
                    (self.config["NUM_STEPS"], self.config["NUM_ENVS"], self.env.num_agents)
                ),
                traj_batch.info,
            )

            metric["update_steps"] = update_step
            metric["loss_info"] = jax.tree_map(lambda x: x.mean(), loss_info)
            # print('returned episode returns', metric["returned_episode_returns"].shape)
            # print('returned episodes', metric["returned_episode"].shape)
            metric["returns"] = metric["returned_episode_returns"].at[:,:,0].get().mean(where=(metric["returned_episode"].at[:, :, 0].get()))
            # TODO add win rate and final win rate

            runner_state = RNNRunnerState(
                update_step + 1, 
                params,
                opt_states,
                env_state,
                last_obs,
                last_done,
                last_episode_done,
                hstates,
                rng,
            )
            
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = RNNRunnerState(
            update_step=0,
            params=params,
            opt_states=opt_states,
            env_state=env_state,
            obs=obsv,
            done=jnp.zeros((self.config["NUM_ACTORS"]), dtype=bool),
            episode_done=jnp.zeros((self.config["NUM_ENVS"]), dtype=bool),
            hstates=init_hstates,
            key=_rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, self.config["NUM_UPDATES"]
        )
        # print('metric returns', metric["returns"])
        return metric["returns"].sum()
        
