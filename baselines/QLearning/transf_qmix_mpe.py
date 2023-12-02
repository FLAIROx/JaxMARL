"""
End-to-End JAX Implementation of QMix.

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

The implementation closely follows the original Pymarl: https://github.com/oxwhirl/pymarl/blob/master/src/learners/q_learner.py
"""
import os
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import NamedTuple, Dict, Union

import chex
import optax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import flashbax as fbx
from flax.core import frozen_dict
import wandb
import hydra
from omegaconf import OmegaConf
from safetensors.flax import save_file
from flax.traverse_util import flatten_dict

from jaxmarl import make
from jaxmarl.wrappers.baselines import CTRolloutManager
from jaxmarl.environments.smax import map_name_to_scenario

from fast_attention import make_fast_generalized_attention


class EncoderBlock(nn.Module):
    hidden_dim : int  # Input dimension is needed here since it is equal to the output dimension (residual connection)
    num_heads : int
    dim_feedforward : int
    dropout_prob : float = 0.

    def setup(self):
        # Attention layer
        raw_attention_fn = make_fast_generalized_attention(
            self.hidden_dim // self.num_heads,
            renormalize_attention=True,
            nb_features=self.hidden_dim,
            unidirectional=False
        )
        self.self_attn = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dropout_rate=self.dropout_prob,
            attention_fn=raw_attention_fn
        )
        # Two-layer MLP
        self.linear = [
            nn.Dense(self.dim_feedforward),
            nn.Dropout(self.dropout_prob),
            nn.relu,
            nn.Dense(self.hidden_dim)
        ]
        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
        self.dropout = nn.Dropout(self.dropout_prob)

    def __call__(self, x, mask=None, deterministic=True):

        # Attention part
        x = self.self_attn(inputs_q=x, inputs_kv=x, deterministic=deterministic)
        x = x + self.dropout(x, deterministic=deterministic)
        x = self.norm1(x)

        # MLP part
        for l in self.linear:
            x = l(x) if not isinstance(l, nn.Dropout) else l(x, deterministic=deterministic)
        x = x + self.dropout(x, deterministic=deterministic)
        x = self.norm2(x)

        return x

class Embedder(nn.Module):
    hidden_dim: int
    init_scale: int
    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.hidden_dim, kernel_init=orthogonal(self.init_scale), bias_init=constant(0.0))(x)
    

class TransformerAgent(nn.Module):
    action_dim: int
    hidden_dim: int
    init_scale: float
    transf_num_layers: int
    transf_num_heads: int
    transf_dim_feedforward: int
    transf_dropout_prob: float
    deterministic: bool

    def setup(self):
        self.embedder = Embedder(self.hidden_dim, init_scale=self.init_scale)
        self.encoders = [
            EncoderBlock(self.hidden_dim, self.transf_num_heads, self.transf_dim_feedforward, self.transf_dropout_prob)
            for _ in range(self.transf_num_layers)
        ]
        self.q_proj = nn.Dense(self.action_dim, kernel_init=orthogonal(self.init_scale), bias_init=constant(0.0))

    def __call__(self, hidden_state, x, embed=True):
        # be careful when you jit this function embed must be concrete
        ins, resets = x
        if embed:
            embeddings = self.embedder(ins)
        else:
            embeddings = ins

        # reset the hidden state if done
        hidden_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(self.hidden_dim, *resets.shape, 1),
            hidden_state,
        )
        # add the reccurent hidden_state to the inputs and pass thorough the transformer encoders
        x = jnp.concatenate((hidden_state, embeddings), axis=-2) # concatenate on the n_entities dim
        for encoder in self.encoders:
            x = encoder(x, mask=None, deterministic=self.deterministic) # TODO: dynamic masking
        hidden_state = x[..., 0:1, :] # retrieve the hidden state from the stack
        
        q_vals = self.q_proj(hidden_state)
        return hidden_state, q_vals

    
    @staticmethod
    def initialize_carry(hidden_size, *batch_size):
        return jnp.zeros((*batch_size, hidden_size))
    

class ScannedTransformerHypernetwork(nn.Module):

    action_dim: int
    hidden_dim: int
    agent_init_scale: float
    agent_transf_num_layers: int
    agent_transf_num_heads: int
    agent_transf_dim_feedforward: int
    agent_transf_dropout_prob: float
    hyper_transf_num_layers: int
    hyper_transf_num_heads: int
    hyper_transf_dim_feedforward: int
    hyper_transf_dropout_prob: float
    deterministic: bool = True

    def setup(self):
        self.agent = TransformerAgent(
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            init_scale=self.agent_init_scale,
            transf_num_layers=self.agent_transf_num_layers,
            transf_num_heads=self.agent_transf_num_heads,
            transf_dim_feedforward=self.agent_transf_dim_feedforward,
            transf_dropout_prob=self.agent_transf_dropout_prob,
            deterministic=self.deterministic,
        )
        self.hypernetwork = [
            EncoderBlock(self.hidden_dim, self.hyper_transf_num_heads, self.hyper_transf_dim_feedforward, self.hyper_transf_dropout_prob)
            for _ in range(self.hyper_transf_num_layers)
        ] # the hypernetwork is a stack of transformer encoders

    @partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    def __call__(self, carry, x):
        hs_agent, hs_mixer = carry
        agent_embedding, mixer_embedding, done, done_global = x

        # agent pass
        hs_agent, q_vals = self.agent(hs_agent, (agent_embedding, done), embed=False)

        # hypernetwork pass
        hs_mixer = jnp.where(
            done_global,
            TransformerAgent.initialize_carry(self.hidden_dim, *done_global.shape, 1),
            hs_mixer
        )
        embeddings = jnp.concatenate((
            hs_mixer,
            mixer_embedding,
            jax.lax.stop_gradient(hs_agent.reshape(-1, done.shape[-1]//done_global.shape[-1], self.hidden_dim))
        ), axis=-2)
        for layer in self.hypernetwork:
            embeddings = layer(embeddings, mask=None, deterministic=self.deterministic)
        hs_mixer = embeddings[..., 0:1, :]
        
        return (hs_agent, hs_mixer), (q_vals, embeddings)
    

class TransformerMixer(nn.Module):

    action_dim: int
    hidden_dim: int
    agent_init_scale: float
    agent_transf_num_layers: int
    agent_transf_num_heads: int
    agent_transf_dim_feedforward: int
    agent_transf_dropout_prob: float
    mixer_init_scale: float
    hyper_transf_num_layers: int
    hyper_transf_num_heads: int
    hyper_transf_dim_feedforward: int
    hyper_transf_dropout_prob: float
    deterministic: bool = True

    def setup(self):
        self.hypernetwork = ScannedTransformerHypernetwork(
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            agent_init_scale=self.agent_init_scale,
            agent_transf_num_layers=self.agent_transf_num_layers,
            agent_transf_num_heads=self.agent_transf_num_heads,
            agent_transf_dim_feedforward=self.agent_transf_dim_feedforward,
            agent_transf_dropout_prob=self.agent_transf_dropout_prob,
            hyper_transf_num_layers=self.hyper_transf_num_layers,
            hyper_transf_num_heads=self.hyper_transf_num_heads,
            hyper_transf_dim_feedforward=self.hyper_transf_dim_feedforward,
            hyper_transf_dropout_prob=self.hyper_transf_dropout_prob,
            deterministic=self.deterministic,
        )
        self.obs_embedder = Embedder(self.hidden_dim, init_scale=self.agent_init_scale)
        self.state_embedder = Embedder(self.hidden_dim, init_scale=self.mixer_init_scale)
        self.b2_proj = nn.Dense(1, kernel_init=orthogonal(self.mixer_init_scale), bias_init=constant(0.0))

    def __call__(self, hs, inputs):

        obs, states, dones, dones_global, q_index = inputs
        time_steps, batch_size = dones_global.shape
        n_agents = dones.shape[-1] // batch_size
        agent_embeddings = self.obs_embedder(obs)
        mixer_embeddings = self.state_embedder(states)

        _, (q_vals, hyp_emb) = self.hypernetwork(hs, (agent_embeddings, mixer_embeddings, dones, dones_global))

        q_vals = q_vals.squeeze(-2) # (time_steps, batch_size*nagents, 1 [removed], n_actions)
        chosen_q_vals = jnp.take_along_axis(q_vals, q_index[...,np.newaxis], axis=-1) # (time_steps, batch_size*n_agents, 1)
        chosen_q_vals = chosen_q_vals.reshape(time_steps, batch_size, n_agents) # (time_steps, batch_size, n_agents)
        
        # hypernetwork
        main_emb = hyp_emb[..., 0:1, :] # (time_steps, batch_size, 1, hidden_dim), the hidden state of mixer for each timestep
        w_1 = jnp.abs(hyp_emb[..., -n_agents:, :]).reshape(time_steps, batch_size, n_agents, self.hidden_dim)
        b_1 = main_emb.reshape(time_steps, batch_size, 1, self.hidden_dim)
        w_2 = jnp.abs(main_emb).reshape(time_steps, batch_size, self.hidden_dim, 1)
        b_2 = self.b2_proj(main_emb).reshape(time_steps, batch_size, 1, 1)
    
        # mix
        hidden = nn.elu(jnp.matmul(chosen_q_vals[:, :, None, :], w_1) + b_1)
        q_tot  = jnp.matmul(hidden, w_2) + b_2

        return q_tot.squeeze(), q_vals.reshape(time_steps, batch_size, n_agents, -1) # (time_steps, batch_size), 



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


def make_train(config, env):

    # custom env wrapper for MPE spread (transform obs vectors in matrices)
    def wrapped_get_obs(self, state):
        # relative position between agents and other entities
        rel_pos = state.p_pos - state.p_pos[:self.num_agents, None, :]
        is_self_feat  = (jnp.arange(self.num_entities) == jnp.arange(self.num_agents)[:, np.newaxis])
        is_agent_feat = jnp.tile(
            jnp.concatenate((jnp.ones(self.num_agents), jnp.zeros(self.num_landmarks))),
            (self.num_agents, 1)
        )
        feats = jnp.concatenate((
            rel_pos,
            is_self_feat[:, :, None],
            is_agent_feat[:, :, None],
        ), axis=2)

        obs = {
            a:feats[i]
            for i, a in enumerate(self.agents)
        }
        
        obs['world_state'] = jnp.concatenate((
            state.p_pos,
            state.p_vel,
            is_agent_feat[0][:, None]
        ), axis=1)
        
        return obs
    type(env).get_obs = wrapped_get_obs
    CTRolloutManager.global_state = lambda self, obs, state: obs['world_state']

    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    
    def train(rng):

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        wrapped_env = CTRolloutManager(env, batch_size=config["NUM_ENVS"], preprocess_obs=False)
        test_env = CTRolloutManager(env, batch_size=config["NUM_TEST_EPISODES"], preprocess_obs=False) # batched env for testing (has different batch size)
        init_obs, env_state = wrapped_env.batch_reset(_rng)
        init_dones = {agent:jnp.zeros((config["NUM_ENVS"]), dtype=bool) for agent in env.agents+['__all__']}

        # INIT BUFFER
        # to initalize the buffer is necessary to sample a trajectory to know its strucutre
        def _env_sample_step(env_state, unused):
            rng, key_a, key_s = jax.random.split(jax.random.PRNGKey(0), 3) # use a dummy rng here
            key_a = jax.random.split(key_a, env.num_agents)
            actions = {agent: wrapped_env.batch_sample(key_a[i], agent) for i, agent in enumerate(env.agents)}
            obs, env_state, rewards, dones, infos = wrapped_env.batch_step(key_s, env_state, actions)
            transition = Transition(obs, actions, rewards, dones)
            return env_state, transition
        _, sample_traj = jax.lax.scan(
            _env_sample_step, env_state, None, config["NUM_STEPS"]
        )
        sample_traj_unbatched = jax.tree_map(lambda x: x[:, 0], sample_traj) # remove the NUM_ENV dim
        buffer = fbx.make_flat_buffer(
            max_length=config['BUFFER_SIZE'],
            min_length=config['BUFFER_BATCH_SIZE'],
            sample_batch_size=config['BUFFER_BATCH_SIZE'],
            add_sequences=True,
            add_batch_size=None,
        )
        buffer_state = buffer.init(sample_traj_unbatched)

        # INIT NETWORK
        # init agent
        init_x = (
            jnp.zeros((1, 1, sample_traj.obs[env.agents[0]].shape[-1])), # (batch_size, n_entities, obs_dim)
            jnp.zeros((1,)) # (batch size)
        )
        init_hs = TransformerAgent.initialize_carry(config['AGENT_HIDDEN_DIM'], 1, 1) # (batch_size, 1, hidden_dim)
        agent = TransformerAgent(
            action_dim=wrapped_env.max_action_space,
            hidden_dim=config['AGENT_HIDDEN_DIM'],
            init_scale=config['AGENT_INIT_SCALE'],
            transf_num_layers=config['AGENT_TRANSF_NUM_LAYERS'],
            transf_num_heads=config['AGENT_TRANSF_NUM_HEADS'],
            transf_dim_feedforward=config['AGENT_TRANSF_DIM_FF'],
            transf_dropout_prob=0.,
            deterministic=True,
        )

        rng, _rng = jax.random.split(rng)
        agent_params = agent.init(_rng, init_hs, init_x)

        # init mixer
        init_x = (
            jnp.zeros((1, 1, sample_traj.obs[env.agents[0]].shape[-1])), # obs: (batch_size, n_entities, obs_dim)
            jnp.zeros((1,)) # (batch size)
        )
        init_hs = TransformerAgent.initialize_carry(config['AGENT_HIDDEN_DIM'], 1, 1) # (batch_size, 1, hidden_dim)
        agent = TransformerAgent(
            action_dim=wrapped_env.max_action_space,
            hidden_dim=config['AGENT_HIDDEN_DIM'],
            init_scale=config['AGENT_INIT_SCALE'],
            transf_num_layers=config['AGENT_TRANSF_NUM_LAYERS'],
            transf_num_heads=config['AGENT_TRANSF_NUM_HEADS'],
            transf_dim_feedforward=config['AGENT_TRANSF_DIM_FF'],
            transf_dropout_prob=0.,
            deterministic=True,
        )

        rng, _rng = jax.random.split(rng)
        agent_params = agent.init(_rng, init_hs, init_x)

        # init mixer
        init_x =(
            jnp.zeros((1, 1, 1, sample_traj.obs[env.agents[0]].shape[-1])), # obs: time_steps, batch_size*n_agents, n_entities, obs_size
            jnp.zeros((1, 1, 1, sample_traj.obs['__all__'].shape[-1])), # states: time_steps, batch_size, n_entities, state_size,
            jnp.zeros((1, 1,)), # agents' dones: time_sptes, batch_size*n_agents,
            jnp.zeros((1, 1,)), # global dones: time_steps, batch_size
            jnp.zeros((1, 1), dtype=int), # q_index: time_steps, batch_size*n_agents
        )
        init_hs = (init_hs, init_hs) # one hidden state for agent, one for mixer (batch_size*n_agent, 1, hidden_dim), (batch_size, 1, hidden_dim)

        mixer = TransformerMixer(
            action_dim=wrapped_env.max_action_space,
            agent_init_scale=config['AGENT_INIT_SCALE'],
            hidden_dim=config['AGENT_HIDDEN_DIM'],
            agent_transf_num_layers=config['AGENT_TRANSF_NUM_LAYERS'],
            agent_transf_num_heads=config['AGENT_TRANSF_NUM_HEADS'],
            agent_transf_dim_feedforward=config['AGENT_TRANSF_DIM_FF'],
            agent_transf_dropout_prob=0.,
            mixer_init_scale=config['MIXER_INIT_SCALE'],
            hyper_transf_num_layers=config['MIXER_TRANSF_NUM_LAYERS'],
            hyper_transf_num_heads=config['MIXER_TRANSF_NUM_HEADS'],
            hyper_transf_dim_feedforward=config['MIXER_TRANSF_DIM_FF'],
            hyper_transf_dropout_prob=0.,
            deterministic=True,
        )

        rng, _rng = jax.random.split(rng)
        mixer_params = mixer.init(_rng, init_hs, init_x)

        # the mixer params contain a copy of the agents params and updates them
        def update_agent_params(agent_params, mixer_params):
            agent_params['params'] = mixer_params['params']['hypernetwork']['agent']
            agent_params['params']['embedder'] = mixer_params['params']['obs_embedder']
            return agent_params
        agent_params = update_agent_params(agent_params, mixer_params)

        # init optimizer
        network_params = {'agent':agent_params, 'mixer':mixer_params}
        def linear_schedule(count):
            frac = 1.0 - (count / config["NUM_UPDATES"])
            return config["LR"] * frac
        lr = linear_schedule if config.get('LR_LINEAR_DECAY', False) else config['LR']
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adamw(learning_rate=lr, eps=config['EPS_ADAM'], weight_decay=config['WEIGHT_DECAY_ADAM']),
        )
        train_state = TrainState.create(
            apply_fn=None,
            params=network_params,
            tx=tx,
        )
        # target network params
        target_network_params = jax.tree_map(lambda x: jnp.copy(x), train_state.params)

        # INIT EXPLORATION STRATEGY
        explorer = EpsilonGreedy(
            start_e=config["EPSILON_START"],
            end_e=config["EPSILON_FINISH"],
            duration=config["EPSILON_ANNEAL_TIME"]
        )
        # print number of params
        agent_params = sum(x.size for x in jax.tree_util.tree_leaves(network_params['agent']))
        mixer_params = sum(x.size for x in jax.tree_util.tree_leaves(network_params['mixer'])) - agent_params
        jax.debug.print("Number of agent params: {x}", x=agent_params)
        jax.debug.print("Number of mixer params: {x}", x=mixer_params)

        # depending if using parameters sharing or not, q-values are computed using one or multiple parameters
        def homogeneous_pass(params, hidden_state, obs, dones):
            # concatenate agents and parallel envs to process them in one batch
            agents, flatten_agents_obs = zip(*obs.items())
            original_shape = flatten_agents_obs[0].shape # assumes obs shape is the same for all agents
            batched_input = (
                jnp.concatenate(flatten_agents_obs, axis=0), # (n_agents*n_envs, n_entities, obs_size)
                jnp.concatenate([dones[agent] for agent in agents], axis=0), # ensure to not pass other keys (like __all__)
            )
            hidden_state, q_vals = agent.apply(params, hidden_state, batched_input)
            q_vals = q_vals.reshape(original_shape[0], len(agents), -1)
            q_vals = {a:q_vals[:,i] for i,a in enumerate(agents)}
            return hidden_state, q_vals


        # TRAINING LOOP
        def _update_step(runner_state, unused):

            train_state, target_network_params, env_state, buffer_state, time_state, init_obs, init_dones, test_metrics, rng = runner_state

            # EPISODE STEP
            def _env_step(step_state, unused):

                params, env_state, last_obs, last_dones, hstate, rng, t = step_state

                # prepare rngs for actions and step
                rng, key_a, key_s = jax.random.split(rng, 3)

                # SELECT ACTION
                # add a dummy time_step dimension to the agent input
                obs_   = {a:last_obs[a] for a in env.agents} # ensure to not pass the global state (obs["__all__"]) to the network
                # get the q_values from the agent netwoek
                hstate, q_vals = homogeneous_pass(params, hstate, obs_, last_dones)
                # remove the dummy time_step dimension and index qs by the valid actions of each agent 
                valid_q_vals = jax.tree_util.tree_map(lambda q, valid_idx: q[..., valid_idx], q_vals, wrapped_env.valid_actions)
                # explore with epsilon greedy_exploration
                actions = explorer.choose_actions(valid_q_vals, t, key_a)

                # STEP ENV
                obs, env_state, rewards, dones, infos = wrapped_env.batch_step(key_s, env_state, actions)
                transition = Transition(last_obs, actions, rewards, dones)

                step_state = (params, env_state, obs, dones, hstate, rng, t+1)
                return step_state, transition


            # prepare the step state and collect the episode trajectory
            rng, _rng = jax.random.split(rng)
            hstate = TransformerAgent.initialize_carry(config['AGENT_HIDDEN_DIM'], len(env.agents)*config["NUM_ENVS"], 1) # (n_agents*n_envs, hs_size, 1)

            step_state = (
                train_state.params['agent'],
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
            buffer_traj_batch = jax.tree_util.tree_map(lambda x:jnp.swapaxes(x, 0, 1), traj_batch) # put the batch size (num envs) in first axis
            buffer_state = buffer.add(buffer_state, buffer_traj_batch)

            # LEARN PHASE
            def batchify(x, axis=1):
                return jnp.concatenate([x[agent] for agent in env.agents], axis=axis)
            
            def _loss_fn(params, target_network_params, init_hstate, learn_traj):

                mixer_input = [
                    batchify(learn_traj.obs), # obs
                    learn_traj.obs['__all__'], # state 
                    batchify(learn_traj.dones), # dones
                    learn_traj.dones['__all__'], # done global
                    batchify(learn_traj.actions), # chosen actions q indexes
                ]
                
                chosen_action_qvals_mix, q_vals = mixer.apply(
                    params['mixer'],
                    init_hstate,
                    mixer_input,
                )
                
                # get the indexes of the greedy actions
                q_vals = {agent:q_vals[:, :, i] for i, agent in enumerate(env.agents)}
                argmax_q_vals = jax.tree_util.tree_map(lambda q, valid_idx: jnp.argmax(q[..., valid_idx], axis=-1), q_vals, wrapped_env.valid_actions)
                mixer_input[-1] = batchify(argmax_q_vals)
                
                target_max_qvals_mix, _ = jax.lax.stop_gradient(mixer.apply(
                    target_network_params['mixer'],
                    init_hstate,
                    mixer_input,
                ))

                chosen_action_qvals_mix = chosen_action_qvals_mix[:-1] # avoid last time step
                target_max_qvals_mix = target_max_qvals_mix[1:] # avoid first time step

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
            learn_traj = buffer.sample(buffer_state, _rng).experience.first # (batch_size, max_time_steps, ...)
            learn_traj = jax.tree_map(lambda x: jnp.swapaxes(x, 0, 1), learn_traj) # (max_time_steps, batch_size, ...)
            hs_agent = TransformerAgent.initialize_carry(config['AGENT_HIDDEN_DIM'], len(env.agents)*config["BUFFER_BATCH_SIZE"], 1) # (n_agents*batch_size, hs_size)
            hs_mixer = TransformerAgent.initialize_carry(config['AGENT_HIDDEN_DIM'], config["BUFFER_BATCH_SIZE"], 1) # (batch_size, hs_size)
            init_hs = (hs_agent, hs_mixer)

            # compute loss and optimize grad
            grad_fn = jax.value_and_grad(_loss_fn, has_aux=False)
            loss, grads = grad_fn(train_state.params, target_network_params, init_hs, learn_traj)
            train_state = train_state.apply_gradients(grads=grads)
            train_state.params['agent'] = update_agent_params(train_state.params['agent'], train_state.params['mixer'])

            # UPDATE THE VARIABLES AND RETURN
            # reset the environment
            rng, _rng = jax.random.split(rng)
            init_obs, env_state = wrapped_env.batch_reset(_rng)
            init_dones = {agent:jnp.zeros((config["NUM_ENVS"]), dtype=bool) for agent in env.agents+['__all__']}

            # update the states
            time_state['timesteps'] = step_state[-1]
            time_state['updates']   = time_state['updates'] + 1

            # update the target network if necessary
            target_network_params = jax.lax.cond(
                time_state['updates'] % config['TARGET_UPDATE_INTERVAL'] == 0,
                lambda _: jax.tree_map(lambda x: jnp.copy(x), train_state.params),
                lambda _: target_network_params,
                operand=None
            )

            # update the greedy rewards
            rng, _rng = jax.random.split(rng)
            test_metrics = jax.lax.cond(
                time_state['updates'] % (config["TEST_INTERVAL"] // config["NUM_STEPS"] // config["NUM_ENVS"]) == 0,
                lambda _: get_greedy_metrics(_rng, train_state.params['agent'], time_state),
                lambda _: test_metrics,
                operand=None
            )

            # update the returning metrics
            metrics = {
                'timesteps': time_state['timesteps']*config['NUM_ENVS'],
                'updates' : time_state['updates'],
                'loss': loss,
                'rewards': jax.tree_util.tree_map(lambda x: jnp.sum(x, axis=0).mean(), traj_batch.rewards)
            }
            metrics.update(test_metrics) # add the test metrics dictionary

            if config.get('WANDB_ONLINE_REPORT', False):
                def callback(metrics):
                    wandb.log(
                        {
                            "returns": metrics['rewards']['__all__'].mean(),
                            "test_returns": metrics['test_returns']['__all__'].mean(),
                            "timestep": metrics['timesteps'],
                            "updates": metrics['updates'],
                            "loss": metrics['loss'],
                        }
                    )
                jax.debug.callback(callback, metrics)

            runner_state = (
                train_state,
                target_network_params,
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
                hstate, q_vals = homogeneous_pass(params, hstate, obs_, last_dones)
                actions = jax.tree_util.tree_map(lambda q, valid_idx: jnp.argmax(q[..., valid_idx], axis=-1), q_vals, wrapped_env.valid_actions)
                obs, env_state, rewards, dones, infos = test_env.batch_step(key_s, env_state, actions)
                step_state = (params, env_state, obs, dones, hstate, rng)
                return step_state, (rewards, dones)
            rng, _rng = jax.random.split(rng)
            init_obs, env_state = test_env.batch_reset(_rng)
            init_dones = {agent:jnp.zeros((config["NUM_TEST_EPISODES"]), dtype=bool) for agent in env.agents+['__all__']}
            rng, _rng = jax.random.split(rng)
            hstate = TransformerAgent.initialize_carry(config['AGENT_HIDDEN_DIM'], len(env.agents)*config["NUM_TEST_EPISODES"], 1) # (n_agents*n_envs, 1, hs_size)

            step_state = (
                params,
                env_state,
                init_obs,
                init_dones,
                hstate, 
                _rng,
            )
            step_state, rews_dones = jax.lax.scan(
                _greedy_env_step, step_state, None, config["NUM_STEPS"]
            )
            # compute the episode returns of the first episode that is done for each parallel env
            def first_episode_returns(rewards, dones):
                first_done = jax.lax.select(jnp.argmax(dones)==0., dones.size, jnp.argmax(dones))
                first_episode_mask = jnp.where(jnp.arange(dones.size) <= first_done, True, False)
                return jnp.where(first_episode_mask, rewards, 0.).sum()
            all_dones = rews_dones[1]['__all__']
            returns = jax.tree_map(lambda r: jax.vmap(first_episode_returns, in_axes=1)(r, all_dones), rews_dones[0])
            metrics = {
                'test_returns': returns # episode returns
            }
            if config.get('VERBOSE', False):
                def callback(timestep, val):
                    print(f"Timestep: {timestep}, return: {val}")
                jax.debug.callback(callback, time_state['timesteps']*config['NUM_ENVS'], returns['__all__'].mean())
            return metrics
        
        time_state = {
            'timesteps':jnp.array(0),
            'updates':  jnp.array(0)
        }
        rng, _rng = jax.random.split(rng)
        test_metrics = get_greedy_metrics(_rng, train_state.params['agent'],time_state) # initial greedy metrics

        # train
        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            target_network_params,
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
    alg_name = f'transf_qmix'
    
    # smac init neeeds a scenario
    if 'SMAC' in env_name:
        config['env']['ENV_KWARGS']['scenario'] = map_name_to_scenario(config['env']['MAP_NAME'])
        env_name = 'jaxmarl_'+config['env']['MAP_NAME']

    env = make(config["env"]["ENV_NAME"], **config['env']['ENV_KWARGS'])
    config["alg"]["NUM_STEPS"] = config["alg"].get("NUM_STEPS", env.max_steps) # default steps defined by the env
    
    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=[
            alg_name.upper(),
            env_name.upper(),
            "TRANSFORMER",
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
    
