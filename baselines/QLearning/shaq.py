"""
End-to-End JAX Implementation of SHAQ.

Notice:
- Agents are controlled by a single RNN architecture.
- You can choose if sharing parameters between agents or not.
- Works also with non-homogenous agents (different obs/action spaces)
- Experience replay is a simple buffer with uniform sampling.
- Uses Double Q-Learning with a target agent network (hard-updated).
- You can use TD Loss (pymarl2) or DDQN loss (pymarl)
- Adam optimizer is used instead of RMSPROP.
- The environment is reset at the end of each episode.
- Trained with a team reward (reward['__all__'])
- At the moment, last_actions are not included in the agents' observations.
- The \hat{\alpha} in the paper is alpha_estimates in the code.
- You can manually choose the alpha of fixed values by manually setting MANUAL_ALPHA_ESTIMATES in configs (to reproduce the experiments in Appendix C.4 of the paper).
- About the setting of LR_ALPHA, you can refer to README of https://github.com/hsvgbkhgbv/shapley-q-learning or the results shown in Appendix C.4 of the paper.
- Right now, we have tested this SHAQ version implemented in JAX on MPE and SMAX.

The implementation closely follows the original SHAQ repo implemented by Pymarl framework: https://github.com/hsvgbkhgbv/shapley-q-learning
"""
import os
import jax
import jax.numpy as jnp
from functools import partial
from typing import NamedTuple, Dict, Union
import numpy as np
import chex
import optax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from flax.core import frozen_dict
import wandb
import hydra
from omegaconf import OmegaConf
import flashbax as fbx
from safetensors.flax import save_file
from flax.traverse_util import flatten_dict


from jaxmarl import make
from jaxmarl.environments.smax import map_name_to_scenario
from jaxmarl.wrappers.baselines import LogWrapper, SMAXLogWrapper, CTRolloutManager

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
        hidden_size = ins.shape[-1]
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(hidden_size, *ins.shape[:-1]),
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
    
class AgentRNN(nn.Module):
    # homogenous agent for parameters sharing, assumes all agents have same obs and action dim
    action_dim: int
    hidden_dim: int
    init_scale: float

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        embedding = nn.Dense(self.hidden_dim, kernel_init=orthogonal(self.init_scale), bias_init=constant(0.0))(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)
        
        q_vals = nn.Dense(self.action_dim, kernel_init=orthogonal(self.init_scale), bias_init=constant(0.0))(embedding)

        return hidden, q_vals
    

class HyperNetwork(nn.Module):
    """HyperNetwork for generating weights of QMix' mixing network."""
    hidden_dim: int
    output_dim: int
    init_scale: float

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(self.init_scale), bias_init=constant(0.))(x)
        x = nn.relu(x)
        x = nn.Dense(self.output_dim, kernel_init=orthogonal(self.init_scale), bias_init=constant(0.))(x)
        return x


class AlphaEstimate(nn.Module):
    """Generate alpha estimate to weight the different types (max or not) of agents' q-values"""
    sample_size: int
    embedding_dim: int
    hypernet_hidden_dim: int
    init_scale: float
    rng_key: jnp.ndarray
    
    def sample_grandcoalitions(self, batch_size, n_agents):
        """
        E.g., batch_size = 2, sample_size = 1, n_agents = 3:
            >>> grand_coalitions_pos
            tensor([[2, 0, 1],
                    [1, 2, 0]])

            >>> subcoalition_map
            tensor([[[[1., 1., 1.],
                        [1., 0., 0.],
                        [1., 1., 0.]]],

                    [[[1., 1., 0.],
                        [1., 1., 1.],
                        [1., 0., 0.]]]])

            >>> individual_map
            tensor([[[[0., 0., 1.],
                        [1., 0., 0.],
                        [0., 1., 0.]]],

                    [[[0., 1., 0.],
                        [0., 0., 1.],
                        [1., 0., 0.]]]])
        """
        seq_set = jnp.tril(jnp.ones((n_agents, n_agents)), 0)
        rng_keys = jax.random.split(self.rng_key, num=batch_size*self.sample_size)
        agent_seq = jnp.arange(0, n_agents)
        vectorized_grand_coalitions_pos_gen = jax.vmap(lambda k: jax.random.choice(rng_keys[k], agent_seq, axis=0, replace=False, shape=(n_agents,)))
        grand_coalitions_pos = vectorized_grand_coalitions_pos_gen(jnp.arange(0, batch_size*self.sample_size)) # shape = (batch_size*sample_size, n_agents)

        grand_coalitions_pos_reshaped = grand_coalitions_pos.reshape(-1, 1)
        individual_map_init = jnp.zeros((batch_size*self.sample_size*n_agents, n_agents))
        individual_map                = individual_map_init.at[(jnp.arange(batch_size*self.sample_size*n_agents), grand_coalitions_pos_reshaped[:, 0])].set(1)
        individual_map                = individual_map.reshape(batch_size, self.sample_size, n_agents, n_agents)
        subcoalition_map              = jnp.matmul(individual_map, seq_set)

        """
        Construct the grand coalition (in sequence by agent_idx) from the grand_coalitions_pos (e.g., pos_idx <- grand_coalitions_pos[agent_idx])
        grand_coalitions = []
        for grand_coalition_pos in grand_coalitions_pos:
            grand_coalition = jnp.zeros_like(grand_coalition_pos)
            for agent, pos in enumerate(grand_coalition_pos):
                grand_coalition[pos] = agent
            grand_coalitions.append(grand_coalition)
        """
        offset                     = (jnp.arange(batch_size*self.sample_size)*n_agents).reshape(-1, 1)
        grand_coalitions_pos_alter = grand_coalitions_pos + offset
        grand_coalitions           = jnp.zeros_like(grand_coalitions_pos_alter.flatten())
        grand_coalitions           = grand_coalitions.at[grand_coalitions_pos_alter.flatten()].set(jnp.arange(batch_size*self.sample_size*n_agents))
        grand_coalitions           = grand_coalitions.reshape(batch_size*self.sample_size, n_agents) - offset
        grand_coalitions           = jnp.tile(
                                        jnp.expand_dims(grand_coalitions, 1), (1, n_agents, 1)
                                     ).reshape(batch_size, self.sample_size, n_agents, n_agents) # shape = (batch_size, sample_size, n_agents, n_agents)

        return subcoalition_map, individual_map, grand_coalitions

    @nn.compact
    def __call__(self, q_vals, states):
        n_agents, time_steps, batch_size = q_vals.shape
        agent_qs = jnp.transpose(q_vals, (1, 2, 0)).reshape(time_steps*batch_size, n_agents)[..., jnp.newaxis] # shape = (time_steps*batch_size, n_agents, 1)

        # get subcoalition map including agent i
        subcoalition_map, individual_map, grand_coalitions = self.sample_grandcoalitions(time_steps*batch_size, n_agents)

        # reshape the grand coalition map for rearranging the sequence of actions of agents, shape = (time_steps*batch_size, sample_size, n_agents, n_agents, 1)
        grand_coalitions      = jnp.tile(jnp.expand_dims(grand_coalitions, -1), (1, 1, 1, 1, 1))

        # remove agent i from the subcloation map, shape = (time_steps*batch_size, sample_size, n_agents, n_agents, 1)
        subcoalition_map_no_i = jnp.tile(jnp.expand_dims(subcoalition_map - individual_map, -1), (1, 1, 1, 1, 1))

        # reshape actions for further process on coalitions, shape = (time_steps*batch_size, sample_size, n_agents, n_agents, 1)
        reshape_agent_qs   = jnp.tile(jnp.expand_dims(agent_qs, (1, 2)), (1, self.sample_size, n_agents, 1, 1))
        reshape_agent_qs   = jnp.take_along_axis(reshape_agent_qs, grand_coalitions, axis=-2)

        # get actions of its coalition memebers for each agent, shape = (time_steps*batch_size, sample_size, n_agents, n_agents, 1)
        agent_qs_coalition = reshape_agent_qs * subcoalition_map_no_i

        # get actions vector of its coalition members for each agent, shape = (time_steps*batch_size, sample_size, n_agents, 1)
        subcoalition_map_no_i_sum   = subcoalition_map_no_i.sum(axis=-2)
        subcoalition_map_no_i_sum   = subcoalition_map_no_i_sum + (subcoalition_map_no_i_sum==0)
        agent_qs_coalition_norm_vec = agent_qs_coalition.sum(axis=-2) / subcoalition_map_no_i_sum

        # get action vector of each agent, shape = (time_steps*batch_size, sample_size, n_agents, 1)
        agent_qs_individual         = jnp.tile(jnp.expand_dims(agent_qs, 1), (1, self.sample_size, 1, 1))

        # preprocess the inputs to neural networks
        reshape_agent_qs_coalition_norm_vec = agent_qs_coalition_norm_vec.reshape(-1, 1) # shape = (time_steps*batch_size*sample_size*n_agents, 1)
        reshape_agent_qs_individual         = agent_qs_individual.reshape(-1, 1) # shape = (time_steps*batch_size*sample_size*n_agents, 1)
        reshape_states                      = jnp.tile(
                                                jnp.expand_dims(states, (1, 2)), (1, self.sample_size, n_agents, 1)
                                            ).reshape(time_steps*batch_size*self.sample_size*n_agents, -1) # shape = (time_steps*batch_size*sample_size*n_agents, state_dim)
        inputs = jnp.concatenate((reshape_agent_qs_coalition_norm_vec, reshape_agent_qs_individual), axis=-1)[:,jnp.newaxis,...] # shape = (time_steps*batch_size*sample_size*n_agents, 1, 2)

        # hypernetwork
        w_1 = HyperNetwork(hidden_dim=self.hypernet_hidden_dim, output_dim=self.embedding_dim*2, init_scale=self.init_scale)(reshape_states)
        b_1 = nn.Dense(self.embedding_dim, kernel_init=orthogonal(self.init_scale), bias_init=constant(0.))(reshape_states)
        w_2 = HyperNetwork(hidden_dim=self.hypernet_hidden_dim, output_dim=self.embedding_dim, init_scale=self.init_scale)(reshape_states)
        b_2 = HyperNetwork(hidden_dim=self.embedding_dim, output_dim=1, init_scale=self.init_scale)(reshape_states)
        
        # monotononicity and reshaping
        w_1 = jnp.abs(w_1.reshape(time_steps*batch_size*self.sample_size*n_agents, 2, self.embedding_dim))
        b_1 = b_1.reshape(time_steps*batch_size*self.sample_size*n_agents, 1, self.embedding_dim)
        w_2 = jnp.abs(w_2.reshape(time_steps*batch_size*self.sample_size*n_agents, self.embedding_dim, 1))
        b_2 = b_2.reshape(time_steps*batch_size*self.sample_size*n_agents, 1, 1)
    
        # mix
        hidden = nn.elu(jnp.matmul(inputs, w_1) + b_1)
        y      = jnp.matmul(hidden, w_2) + b_2

        # reshape, shape = (time_steps, batch_size, sample_size, n_agents)
        alpha_estimates = jnp.abs(y.squeeze()).reshape(time_steps, batch_size, self.sample_size, n_agents)
        # normalise over the sample_size, shape = (time_steps, batch_size, n_agents)
        alpha_estimates = jnp.transpose(alpha_estimates.mean(axis=2), (2, 0, 1))

        return alpha_estimates


class SHAQMixer(nn.Module):
    sample_size: int
    embedding_dim: int
    hypernet_hidden_dim: int
    init_scale: float
    rng_key: jnp.ndarray

    @nn.compact
    def __call__(self, q_vals, states, max_filter, target, manual_alpha_estimates=None):
        # shape of q_vals, max_filter: (n_agents, time_steps, batch_size)
        n_agents, time_steps, batch_size = q_vals.shape
        if target:
            return jnp.sum(q_vals, axis=0)
        else:
            if manual_alpha_estimates == None:
                alpha_estimates = AlphaEstimate(
                                        sample_size=self.sample_size, 
                                        embedding_dim=self.embedding_dim, 
                                        hypernet_hidden_dim=self.hypernet_hidden_dim, 
                                        init_scale=self.init_scale,
                                        rng_key=self.rng_key
                                    )(q_vals, states)
                # restrict the range of alpha to [1, \infty)
                alpha_estimates = alpha_estimates + 1.
            else:
                alpha_estimates = manual_alpha_estimates * jnp.ones_like(max_filter)
            # agent with non-max action will be given 1
            non_max_filter = 1 - max_filter
            # if the agent with the max-action then alpha = 1. Otherwise, the agent will use the learned alpha
            return jnp.sum((alpha_estimates * non_max_filter + max_filter) * q_vals, axis=0)

class EpsilonGreedy:
    """Epsilon Greedy action selection"""

    def __init__(self, start_e: float, end_e: float, duration: int):
        self.start_e  = start_e
        self.end_e    = end_e
        self.duration = duration
        self.slope    = (end_e - start_e) / duration
        
    @partial(jax.jit, static_argnums=0)
    def get_epsilon(self, t: int):
        e = self.slope*t + self.start_e
        return jnp.clip(e, self.end_e)
    
    @partial(jax.jit, static_argnums=0)
    def choose_actions(self, q_vals: dict, t: int, rng: chex.PRNGKey):
        
        def explore(q, eps, key):
            key_a, key_e   = jax.random.split(key, 2) # a key for sampling random actions and one for picking
            greedy_actions = jnp.argmax(q, axis=-1) # get the greedy actions 
            random_actions = jax.random.randint(key_a, shape=greedy_actions.shape, minval=0, maxval=q.shape[-1]) # sample random actions
            pick_random    = jax.random.uniform(key_e, greedy_actions.shape)<eps # pick which actions should be random
            chosed_actions = jnp.where(pick_random, random_actions, greedy_actions)
            return chosed_actions
        
        eps = self.get_epsilon(t)
        keys = dict(zip(q_vals.keys(), jax.random.split(rng, len(q_vals)))) # get a key for each agent
        chosen_actions = jax.tree_map(lambda q, k: explore(q, eps, k), q_vals, keys)
        return chosen_actions

class Transition(NamedTuple):
    obs: dict
    actions: dict
    rewards: dict
    dones: dict
    infos: dict

def make_train(config, env):

    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    
    def train(rng):

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        wrapped_env = CTRolloutManager(env, batch_size=config["NUM_ENVS"])
        test_env = CTRolloutManager(env, batch_size=config["NUM_TEST_EPISODES"]) # batched env for testing (has different batch size)
        init_obs, env_state = wrapped_env.batch_reset(_rng)
        init_dones = {agent:jnp.zeros((config["NUM_ENVS"]), dtype=bool) for agent in env.agents+['__all__']}

        # INIT BUFFER
        # to initalize the buffer is necessary to sample a trajectory to know its strucutre
        def _env_sample_step(env_state, unused):
            rng, key_a, key_s = jax.random.split(jax.random.PRNGKey(0), 3) # use a dummy rng here
            key_a = jax.random.split(key_a, env.num_agents)
            actions = {agent: wrapped_env.batch_sample(key_a[i], agent) for i, agent in enumerate(env.agents)}
            obs, env_state, rewards, dones, infos = wrapped_env.batch_step(key_s, env_state, actions)
            transition = Transition(obs, actions, rewards, dones, infos)
            return env_state, transition
        _, sample_traj = jax.lax.scan(
            _env_sample_step, env_state, None, config["NUM_STEPS"]
        )
        sample_traj_unbatched = jax.tree_map(lambda x: x[:, 0], sample_traj) # remove the NUM_ENV dim
        buffer = fbx.make_trajectory_buffer(
            max_length_time_axis=config['BUFFER_SIZE']//config['NUM_ENVS'],
            min_length_time_axis=config['BUFFER_BATCH_SIZE'],
            sample_batch_size=config['BUFFER_BATCH_SIZE'],
            add_batch_size=config['NUM_ENVS'],
            sample_sequence_length=1,
            period=1,
        )
        buffer_state = buffer.init(sample_traj_unbatched) 


        # INIT NETWORK
        # init agent
        agent = AgentRNN(action_dim=wrapped_env.max_action_space, hidden_dim=config["AGENT_HIDDEN_DIM"], init_scale=config['AGENT_INIT_SCALE'])
        rng, _rng = jax.random.split(rng)
        if config["PARAMETERS_SHARING"]:
            init_x = (
                jnp.zeros((1, 1, wrapped_env.obs_size)), # (time_step, batch_size, obs_size)
                jnp.zeros((1, 1)) # (time_step, batch size)
            )
            init_hs = ScannedRNN.initialize_carry(config['AGENT_HIDDEN_DIM'], 1) # (batch_size, hidden_dim)
            agent_params = agent.init(_rng, init_hs, init_x)
        else:
            init_x = (
                jnp.zeros((len(env.agents), 1, 1, wrapped_env.obs_size)), # (time_step, batch_size, obs_size)
                jnp.zeros((len(env.agents), 1, 1)) # (time_step, batch size)
            )
            init_hs = ScannedRNN.initialize_carry(config['AGENT_HIDDEN_DIM'], len(env.agents),  1) # (n_agents, batch_size, hidden_dim)
            rngs = jax.random.split(_rng, len(env.agents)) # a random init for each agent
            agent_params = jax.vmap(agent.init, in_axes=(0, 0, 0))(rngs, init_hs, init_x)

        # init mixer
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros((len(env.agents), 1, 1))
        state_size = sample_traj.obs['__all__'].shape[-1]  # get the state shape from the buffer
        init_state = jnp.zeros((1, 1, state_size))
        mixer = SHAQMixer(config['SAMPLE_SIZE'], config['MIXER_EMBEDDING_DIM'], config["MIXER_HYPERNET_HIDDEN_DIM"], config['MIXER_INIT_SCALE'], rng)
        mixer_params = mixer.init(_rng, init_x, init_state, init_x, False, config["MANUAL_ALPHA_ESTIMATES"])

        # init optimizer
        def linear_schedule(count):
            frac = 1.0 - (count / config["NUM_UPDATES"])
            return config["LR"] * frac
        lr       = linear_schedule if config.get('LR_LINEAR_DECAY', False) else config['LR']
        lr_alpha = linear_schedule if config.get('LR_LINEAR_DECAY', False) else config['LR_ALPHA']
        tx       = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adamw(learning_rate=lr, eps=config['EPS_ADAM'], weight_decay=config['WEIGHT_DECAY_ADAM']),
        )
        tx_alpha = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adamw(learning_rate=lr_alpha, eps=config['EPS_ADAM'], weight_decay=config['WEIGHT_DECAY_ADAM']),
        )
        train_state_agent = TrainState.create(
            apply_fn=None,
            params=agent_params,
            tx=tx,
        )
        train_state_mixer = TrainState.create(
            apply_fn=None,
            params=mixer_params,
            tx=tx_alpha,
        )

        # target network params
        target_network_params_agent = jax.tree_map(lambda x: jnp.copy(x), train_state_agent.params)
        target_network_params_mixer = jax.tree_map(lambda x: jnp.copy(x), train_state_mixer.params)

        # INIT EXPLORATION STRATEGY
        explorer = EpsilonGreedy(
            start_e=config["EPSILON_START"],
            end_e=config["EPSILON_FINISH"],
            duration=config["EPSILON_ANNEAL_TIME"]
        )

        # depending if using parameters sharing or not, q-values are computed using one or multiple parameters
        if config["PARAMETERS_SHARING"]:
            def homogeneous_pass(params, hidden_state, obs, dones):
                # concatenate agents and parallel envs to process them in one batch
                agents, flatten_agents_obs = zip(*obs.items())
                original_shape = flatten_agents_obs[0].shape # assumes obs shape is the same for all agents
                batched_input = (
                    jnp.concatenate(flatten_agents_obs, axis=1), # (time_step, n_agents*n_envs, obs_size)
                    jnp.concatenate([dones[agent] for agent in agents], axis=1), # ensure to not pass other keys (like __all__)
                )
                hidden_state, q_vals = agent.apply(params, hidden_state, batched_input)
                q_vals = q_vals.reshape(original_shape[0], len(agents), *original_shape[1:-1], -1) # (time_steps, n_agents, n_envs, action_dim)
                q_vals = {a:q_vals[:,i] for i,a in enumerate(agents)}
                return hidden_state, q_vals
        else:
            def homogeneous_pass(params, hidden_state, obs, dones):
                # homogeneous pass vmapped in respect to the agents parameters (i.e., no parameter sharing)
                agents, flatten_agents_obs = zip(*obs.items())
                batched_input = (
                    jnp.stack(flatten_agents_obs, axis=0), # (n_agents, time_step, n_envs, obs_size)
                    jnp.stack([dones[agent] for agent in agents], axis=0), # ensure to not pass other keys (like __all__)
                )
                # computes the q_vals with the params of each agent separately by vmapping
                hidden_state, q_vals = jax.vmap(agent.apply, in_axes=0)(params, hidden_state, batched_input)
                q_vals = {a:q_vals[i] for i,a in enumerate(agents)}
                return hidden_state, q_vals


        # TRAINING LOOP
        def _update_step(runner_state, unused):

            train_state_agent, train_state_mixer, target_network_params_agent, target_network_params_mixer, env_state, buffer_state, time_state, init_obs, init_dones, test_metrics, rng = runner_state

            # EPISODE STEP
            def _env_step(step_state, unused):

                params, env_state, last_obs, last_dones, hstate, rng, t = step_state

                # prepare rngs for actions and step
                rng, key_a, key_s = jax.random.split(rng, 3)

                # SELECT ACTION
                # add a dummy time_step dimension to the agent input
                obs_   = {a:last_obs[a] for a in env.agents} # ensure to not pass the global state (obs["__all__"]) to the network
                obs_   = jax.tree_map(lambda x: x[np.newaxis, :], obs_)
                dones_ = jax.tree_map(lambda x: x[np.newaxis, :], last_dones)
                # get the q_values from the agent network
                hstate, q_vals = homogeneous_pass(params, hstate, obs_, dones_)
                # remove the dummy time_step dimension and index qs by the valid actions of each agent 
                valid_q_vals = jax.tree_util.tree_map(lambda q, valid_idx: q.squeeze(0)[..., valid_idx], q_vals, wrapped_env.valid_actions)
                # explore with epsilon greedy_exploration
                actions = explorer.choose_actions(valid_q_vals, t, key_a)

                # STEP ENV
                obs, env_state, rewards, dones, infos = wrapped_env.batch_step(key_s, env_state, actions)
                transition = Transition(last_obs, actions, rewards, dones, infos)

                step_state = (params, env_state, obs, dones, hstate, rng, t+1)
                return step_state, transition


            # prepare the step state and collect the episode trajectory
            rng, _rng = jax.random.split(rng)
            if config["PARAMETERS_SHARING"]:
                hstate = ScannedRNN.initialize_carry(config['AGENT_HIDDEN_DIM'], len(env.agents)*config["NUM_ENVS"]) # (n_agents*n_envs, hs_size)
            else:
                hstate = ScannedRNN.initialize_carry(config['AGENT_HIDDEN_DIM'], len(env.agents), config["NUM_ENVS"]) # (n_agents, n_envs, hs_size)

            step_state = (
                train_state_agent.params,
                env_state,
                init_obs,
                init_dones,
                hstate, 
                _rng,
                time_state['timesteps'] # t is needed to compute epsilon
            )

            step_state, traj_batch = jax.lax.scan(
                _env_step, step_state, None, config["NUM_STEPS"]
            )

            # BUFFER UPDATE: save the collected trajectory in the buffer
            buffer_traj_batch = jax.tree_util.tree_map(
                lambda x:jnp.swapaxes(x, 0, 1)[:, np.newaxis], # put the batch dim first and add a dummy sequence dim
                traj_batch
            ) # (num_envs, 1, time_steps, ...)
            buffer_state = buffer.add(buffer_state, buffer_traj_batch)

            # LEARN PHASE
            def q_of_action(q, u):
                """index the q_values with action indices"""
                q_u = jnp.take_along_axis(q, jnp.expand_dims(u, axis=-1), axis=-1)
                return jnp.squeeze(q_u, axis=-1)

            def get_max_filter(q, u):
                max_u = jnp.argmax(q, axis=-1, keepdims=True)
                max_filter = jnp.squeeze(jnp.expand_dims(u, axis=-1)==max_u, axis=-1)
                return max_filter

            def _loss_fn(params_agent, params_mixer, target_network_params_agent, target_network_params_mixer, init_hstate, learn_traj):

                obs_ = {a:learn_traj.obs[a] for a in env.agents} # ensure to not pass the global state (obs["__all__"]) to the network
                _, q_vals = homogeneous_pass(params_agent, init_hstate, obs_, learn_traj.dones)
                _, target_q_vals = homogeneous_pass(target_network_params_agent, init_hstate, obs_, learn_traj.dones)

                # get the q_vals of the taken actions (with exploration) for each agent
                chosen_action_qvals = jax.tree_map(
                    lambda q, u: q_of_action(q, u)[:-1], # avoid last timestep
                    q_vals,
                    learn_traj.actions
                )

                # get the target q value of the greedy actions for each agent
                valid_q_vals = jax.tree_util.tree_map(lambda q, valid_idx: q[..., valid_idx], q_vals, wrapped_env.valid_actions)
                target_max_qvals = jax.tree_map(
                    lambda t_q, q: q_of_action(t_q, jnp.argmax(q, axis=-1))[1:], # avoid first timestep
                    target_q_vals,
                    jax.lax.stop_gradient(valid_q_vals)
                )

                # get the max_filters
                max_filters = jax.tree_map(
                    lambda q, u: get_max_filter(q, u)[:-1],
                    q_vals,
                    learn_traj.actions
                )

                # compute q_tot with the mixer network
                chosen_action_qvals_mix = mixer.apply(
                    params_mixer,
                    jnp.stack(list(chosen_action_qvals.values())),
                    learn_traj.obs['__all__'][:-1],
                    jnp.stack(list(max_filters.values())),
                    False,
                    config["MANUAL_ALPHA_ESTIMATES"]
                )

                target_max_qvals_mix = mixer.apply(
                    target_network_params_mixer,
                    jnp.stack(list(target_max_qvals.values())),
                    learn_traj.obs['__all__'][1:],
                    jnp.stack(list(max_filters.values())),
                    True,
                    config["MANUAL_ALPHA_ESTIMATES"]
                )

                # compute target
                if config.get('TD_LAMBDA_LOSS', True):
                    # time difference loss
                    def _td_lambda_target(ret, values):
                        reward, done, target_qs = values
                        ret = jnp.where(
                            done,
                            target_qs,
                            ret*config['TD_LAMBDA']*config['GAMMA']
                            + reward
                            + (1-config['TD_LAMBDA'])*config['GAMMA']*(1-done)*target_qs
                        )
                        return ret, ret

                    ret = target_max_qvals_mix[-1] * (1-learn_traj.dones['__all__'][-1])
                    ret, td_targets = jax.lax.scan(
                        _td_lambda_target,
                        ret,
                        (learn_traj.rewards['__all__'][-2::-1], learn_traj.dones['__all__'][-2::-1], target_max_qvals_mix[-1::-1])
                    )
                    targets = td_targets[::-1]
                    loss = jnp.mean(0.5*((chosen_action_qvals_mix - jax.lax.stop_gradient(targets))**2))
                else:
                    # standard DQN loss
                    targets = (
                        learn_traj.rewards['__all__'][:-1]
                        + config['GAMMA']*(1-learn_traj.dones['__all__'][:-1])*target_max_qvals_mix
                    )
                    loss = jnp.mean((chosen_action_qvals_mix - jax.lax.stop_gradient(targets))**2)
                
                return loss


            # sample a batched trajectory from the buffer and set the time step dim in first axis
            rng, _rng = jax.random.split(rng)
            learn_traj = buffer.sample(buffer_state, _rng).experience # (batch_size, 1, max_time_steps, ...)
            learn_traj = jax.tree_map(
                lambda x: jnp.swapaxes(x[:, 0], 0, 1), # remove the dummy sequence dim (1) and swap batch and temporal dims
                learn_traj
            ) # (max_time_steps, batch_size, ...)
            if config["PARAMETERS_SHARING"]:
                init_hs = ScannedRNN.initialize_carry(config['AGENT_HIDDEN_DIM'], len(env.agents)*config["BUFFER_BATCH_SIZE"]) # (n_agents*batch_size, hs_size)
            else:
                init_hs = ScannedRNN.initialize_carry(config['AGENT_HIDDEN_DIM'], len(env.agents), config["BUFFER_BATCH_SIZE"]) # (n_agents, batch_size, hs_size)

            # compute loss and optimize grad
            grad_fn = jax.value_and_grad(_loss_fn, argnums=(0,1), has_aux=False)
            loss, (grads_agent, grads_mixer) = grad_fn(train_state_agent.params, train_state_mixer.params, target_network_params_agent, target_network_params_mixer, init_hs, learn_traj)
            train_state_agent = train_state_agent.apply_gradients(grads=grads_agent)
            train_state_mixer = train_state_mixer.apply_gradients(grads=grads_mixer)


            # UPDATE THE VARIABLES AND RETURN
            # reset the environment
            rng, _rng = jax.random.split(rng)
            init_obs, env_state = wrapped_env.batch_reset(_rng)
            init_dones = {agent:jnp.zeros((config["NUM_ENVS"]), dtype=bool) for agent in env.agents+['__all__']}

            # update the states
            time_state['timesteps'] = step_state[-1]
            time_state['updates']   = time_state['updates'] + 1

            # update the target network if necessary
            target_network_params_agent = jax.lax.cond(
                time_state['updates'] % config['TARGET_UPDATE_INTERVAL'] == 0,
                lambda _: jax.tree_map(lambda x: jnp.copy(x), train_state_agent.params),
                lambda _: target_network_params_agent,
                operand=None
            )
            target_network_params_mixer = jax.lax.cond(
                time_state['updates'] % config['TARGET_UPDATE_INTERVAL'] == 0,
                lambda _: jax.tree_map(lambda x: jnp.copy(x), train_state_mixer.params),
                lambda _: target_network_params_mixer,
                operand=None
            )

            # update the greedy rewards
            rng, _rng = jax.random.split(rng)
            test_metrics = jax.lax.cond(
                time_state['updates'] % (config["TEST_INTERVAL"] // config["NUM_STEPS"] // config["NUM_ENVS"]) == 0,
                lambda _: get_greedy_metrics(_rng, train_state_agent.params, time_state),
                lambda _: test_metrics,
                operand=None
            )

            # update the returning metrics
            metrics = {
                'timesteps': time_state['timesteps']*config['NUM_ENVS'],
                'updates' : time_state['updates'],
                'loss': loss,
                'rewards': jax.tree_util.tree_map(lambda x: jnp.sum(x, axis=0).mean(), traj_batch.rewards),
                'eps': explorer.get_epsilon(time_state['timesteps'])
            }
            metrics['test_metrics'] = test_metrics # add the test metrics dictionary

            if config.get('WANDB_ONLINE_REPORT', False):
                def callback(metrics, infos):
                    info_metrics = {
                        k:v[...,0][infos["returned_episode"][..., 0]].mean()
                        for k,v in infos.items() if k!="returned_episode"
                    }
                    wandb.log(
                        {
                            "returns": metrics['rewards']['__all__'].mean(),
                            "timestep": metrics['timesteps'],
                            "updates": metrics['updates'],
                            "loss": metrics['loss'],
                            'epsilon': metrics['eps'],
                            **info_metrics,
                            **{k:v.mean() for k, v in metrics['test_metrics'].items()}
                        }
                    )
                jax.debug.callback(callback, metrics, traj_batch.infos)

            runner_state = (
                train_state_agent,
                train_state_mixer,
                target_network_params_agent,
                target_network_params_mixer,
                env_state,
                buffer_state,
                time_state,
                init_obs,
                init_dones,
                test_metrics,
                rng
            )

            return runner_state, metrics

        def get_greedy_metrics(rng, params, time_state):
            """Help function to test greedy policy during training"""
            def _greedy_env_step(step_state, unused):
                params, env_state, last_obs, last_dones, hstate, rng = step_state
                rng, key_s = jax.random.split(rng)
                obs_   = {a:last_obs[a] for a in env.agents}
                obs_   = jax.tree_map(lambda x: x[np.newaxis, :], obs_)
                dones_ = jax.tree_map(lambda x: x[np.newaxis, :], last_dones)
                hstate, q_vals = homogeneous_pass(params, hstate, obs_, dones_)
                actions = jax.tree_util.tree_map(lambda q, valid_idx: jnp.argmax(q.squeeze(0)[..., valid_idx], axis=-1), q_vals, test_env.valid_actions)
                obs, env_state, rewards, dones, infos = test_env.batch_step(key_s, env_state, actions)
                step_state = (params, env_state, obs, dones, hstate, rng)
                return step_state, (rewards, dones, infos)
            rng, _rng = jax.random.split(rng)
            init_obs, env_state = test_env.batch_reset(_rng)
            init_dones = {agent:jnp.zeros((config["NUM_TEST_EPISODES"]), dtype=bool) for agent in env.agents+['__all__']}
            rng, _rng = jax.random.split(rng)
            if config["PARAMETERS_SHARING"]:
                hstate = ScannedRNN.initialize_carry(config['AGENT_HIDDEN_DIM'], len(env.agents)*config["NUM_TEST_EPISODES"]) # (n_agents*n_envs, hs_size)
            else:
                hstate = ScannedRNN.initialize_carry(config['AGENT_HIDDEN_DIM'], len(env.agents), config["NUM_TEST_EPISODES"]) # (n_agents, n_envs, hs_size)
            step_state = (
                params,
                env_state,
                init_obs,
                init_dones,
                hstate, 
                _rng,
            )
            step_state, (rewards, dones, infos) = jax.lax.scan(
                _greedy_env_step, step_state, None, config["NUM_STEPS"]
            )
            # compute the metrics of the first episode that is done for each parallel env
            def first_episode_returns(rewards, dones):
                first_done = jax.lax.select(jnp.argmax(dones)==0., dones.size, jnp.argmax(dones))
                first_episode_mask = jnp.where(jnp.arange(dones.size) <= first_done, True, False)
                return jnp.where(first_episode_mask, rewards, 0.).sum()
            all_dones = dones['__all__']
            first_returns = jax.tree_map(lambda r: jax.vmap(first_episode_returns, in_axes=1)(r, all_dones), rewards)
            first_infos   = jax.tree_map(lambda i: jax.vmap(first_episode_returns, in_axes=1)(i[..., 0], all_dones), infos)
            metrics = {
                'test_returns': first_returns['__all__'],# episode returns
                **{'test_'+k:v for k,v in first_infos.items()}
            }
            if config.get('VERBOSE', False):
                def callback(timestep, val):
                    print(f"Timestep: {timestep}, return: {val}")
                jax.debug.callback(callback, time_state['timesteps']*config['NUM_ENVS'], first_returns['__all__'].mean())
            return metrics
        
        time_state = {
            'timesteps':jnp.array(0),
            'updates':  jnp.array(0)
        }
        rng, _rng = jax.random.split(rng)
        test_metrics = get_greedy_metrics(_rng, train_state_agent.params, time_state) # initial greedy metrics

        # train
        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state_agent,
            train_state_mixer,
            target_network_params_agent,
            target_network_params_mixer,
            env_state,
            buffer_state,
            time_state,
            init_obs,
            init_dones,
            test_metrics,
            _rng
        )
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {'runner_state':runner_state, 'metrics':metrics}
    
    return train

@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(config):
    config = OmegaConf.to_container(config)

    print('Config:\n', OmegaConf.to_yaml(config))

    env_name = config["env"]["ENV_NAME"]
    alg_name = f'shaq_{"ps" if config["alg"].get("PARAMETERS_SHARING", True) else "ns"}'
    
    # smac init neeeds a scenario
    if 'smax' in env_name.lower():
        config['env']['ENV_KWARGS']['scenario'] = map_name_to_scenario(config['env']['MAP_NAME'])
        env_name = f"{config['env']['ENV_NAME']}_{config['env']['MAP_NAME']}"
        env = make(config["env"]["ENV_NAME"], **config['env']['ENV_KWARGS'])
        env = SMAXLogWrapper(env)
    else:
        env = make(config["env"]["ENV_NAME"], **config['env']['ENV_KWARGS'])
        env = LogWrapper(env)

    env = make(config["env"]["ENV_NAME"], **config['env']['ENV_KWARGS'])
    config["alg"]["NUM_STEPS"] = config["alg"].get("NUM_STEPS", env.max_steps) # default steps defined by the env
    
    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=[
            alg_name.upper(),
            env_name.upper(),
            "RNN",
            "TD_LOSS" if config["alg"].get("TD_LAMBDA_LOSS", True) else "DQN_LOSS",
            f"jax_{jax.__version__}",
        ],
        name=f'{alg_name}_{env_name}',
        config=config,
        mode=config["WANDB_MODE"],
    )
    
    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_vjit = jax.jit(jax.vmap(make_train(config["alg"], env)))
    outs = jax.block_until_ready(train_vjit(rngs))
    
    # save params
    if config['SAVE_PATH'] is not None:
        
        def save_params(params: Dict, filename: Union[str, os.PathLike]) -> None:
            flattened_dict = flatten_dict(params, sep=',')
            save_file(flattened_dict, filename)
            
        model_state = outs['runner_state'][0]
        params = jax.tree_map(lambda x: x[0], model_state.params) # save only params of the firt run
        save_dir = os.path.join(config['SAVE_PATH'], env_name)
        os.makedirs(save_dir, exist_ok=True)
        save_params(params, f'{save_dir}/{alg_name}.safetensors')
        print(f'Parameters of first batch saved in {save_dir}/{alg_name}.safetensors')


if __name__ == "__main__":
    main()
    
