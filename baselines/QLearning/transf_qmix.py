"""
End-to-End JAX Implementation of TransfQMix.

The implementation closely follows the original one https://github.com/mttga/pymarl_transformers with some additional features:
- The embeddings can be normalized with batch norm in order to stabilize the self-attention gradients.
- It's added the possibility to perform $n$ training updates of the network at each update step. 

Currently supports only MPE_spread and SMAX. Remember that to use the transformers in your environment you need 
to reshape the observations and states to matrices. See: jaxmarl.wrappers.transformers
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
import wandb
import hydra
from omegaconf import OmegaConf
from safetensors.flax import save_file
from flax.traverse_util import flatten_dict

from jaxmarl import make
from jaxmarl.wrappers.baselines import LogWrapper, SMAXLogWrapper
from jaxmarl.wrappers.transformers import TransformersCTRolloutManager
from jaxmarl.environments.smax import map_name_to_scenario
from jaxmarl.environments.overcooked import overcooked_layouts 

from typing import Any


class EncoderBlock(nn.Module):
    hidden_dim : int  # Input dimension is needed here since it is equal to the output dimension (residual connection)
    num_heads : int
    dim_feedforward : int
    init_scale: float
    use_fast_attention: bool
    dropout_prob : float = 0.

    def setup(self):
        # Attention layer
        if self.use_fast_attention:
            from utils.fast_attention import make_fast_generalized_attention
            raw_attention_fn = make_fast_generalized_attention(
                self.hidden_dim // self.num_heads,
                renormalize_attention=True,
                nb_features=self.hidden_dim,
                unidirectional=False
            )
            self.self_attn = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                dropout_rate=self.dropout_prob,
                attention_fn=raw_attention_fn,
                kernel_init=nn.initializers.xavier_uniform(),
                use_bias=False,
            )
        else:
            self.self_attn = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                dropout_rate=self.dropout_prob,
                kernel_init=nn.initializers.xavier_uniform(),
                use_bias=False,
            )
        # Two-layer MLP
        self.linear = [
            nn.Dense(self.dim_feedforward, kernel_init=nn.initializers.xavier_uniform(), bias_init=constant(0.0)),
            nn.Dense(self.hidden_dim, kernel_init=nn.initializers.xavier_uniform(), bias_init=constant(0.0))
        ]
        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
        self.dropout = nn.Dropout(self.dropout_prob)

    def __call__(self, x, mask=None, deterministic=True):

        # Attention part
        if mask is not None and not self.use_fast_attention: # masking is not compatible with fast self attention
            mask = jnp.repeat(nn.make_attention_mask(mask, mask), self.num_heads, axis=-3)
        attended = self.self_attn(inputs_q=x, inputs_kv=x, mask=mask, deterministic=deterministic)

        x = self.norm1(attended + x)
        x = x + self.dropout(x, deterministic=deterministic)

        # MLP part
        feedforward = self.linear[0](x)
        feedforward = nn.relu(feedforward)
        feedforward = self.linear[1](feedforward)

        x = self.norm2(feedforward+x)
        x = x + self.dropout(x, deterministic=deterministic)

        return x


class Embedder(nn.Module):
    hidden_dim: int
    init_scale: float
    scale_inputs: bool = True
    activation: bool = False
    @nn.compact
    def __call__(self, x, train:bool):
        if self.scale_inputs:
            x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(self.init_scale), bias_init=constant(0.0))(x)
        if self.activation:
            x = nn.relu(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        return x

class ScannedTransformer(nn.Module):
    
    hidden_dim: int
    init_scale: float
    transf_num_layers: int
    transf_num_heads: int
    transf_dim_feedforward: int
    transf_dropout_prob: float = 0
    deterministic: bool = True
    return_embeddings: bool = False
    use_fast_attention: bool = False

    def setup(self):
        self.encoders = [
            EncoderBlock(
                self.hidden_dim,
                self.transf_num_heads,
                self.transf_dim_feedforward,
                self.init_scale,
                self.use_fast_attention,
                self.transf_dropout_prob,
            ) for _ in range(self.transf_num_layers)
        ]

    @partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    def __call__(self, carry, x):
        hs = carry
        embeddings, mask, done = x

        hs = jnp.where(
            done[:, np.newaxis, np.newaxis],
            self.initialize_carry(self.hidden_dim, *done.shape, 1),
            hs
        )
        embeddings = jnp.concatenate((
            hs,
            embeddings,
        ), axis=-2)
        for layer in self.encoders:
            embeddings = layer(embeddings, mask=mask, deterministic=self.deterministic)
        hs = embeddings[..., 0:1, :]

        # as y return the entire embeddings if required (i.e. transformer mixer), otherwise only agents' hs embeddings
        if self.return_embeddings:
            return hs, embeddings
        else:
            return hs, hs

    @staticmethod
    def initialize_carry(hidden_size, *batch_size):
        return jnp.zeros((*batch_size, hidden_size))
    

class TransformerAgent(nn.Module):
    action_dim: int
    hidden_dim: int
    init_scale_emb: float
    init_scale_transf: float
    init_scale_q: float
    transf_num_layers: int
    transf_num_heads: int
    transf_dim_feedforward: int
    transf_dropout_prob: float
    deterministic: bool
    use_fast_attention: bool = True
    scale_inputs: bool = True
    relu_emb: bool = True

    @nn.compact
    def __call__(self, hs, x, train=True, return_all_hs=False):
        
        ins, resets = x
        embeddings = Embedder(
            self.hidden_dim,
            init_scale=self.init_scale_emb,
            scale_inputs=self.scale_inputs,
            activation=self.relu_emb,
        )(ins, train)
        last_hs, hidden_states = ScannedTransformer(
            hidden_dim=self.hidden_dim,
            init_scale=self.init_scale_transf,
            transf_num_layers=self.transf_num_layers,
            transf_num_heads=self.transf_num_heads,
            transf_dim_feedforward=self.transf_dim_feedforward,
            use_fast_attention=self.use_fast_attention,
            deterministic=True,
            return_embeddings=False,
        )(hs, (embeddings, None, resets))
        q_vals = nn.Dense(self.action_dim, kernel_init=orthogonal(self.init_scale_q), bias_init=constant(0.0))(hidden_states)

        if return_all_hs:
            return last_hs, (hidden_states, q_vals)
        else:
            return last_hs, q_vals
        

class TransformerAgentSmax(nn.Module):
    # variation of transformer agent which uses policy decomposition to
    # compute the q-values of attacking an enemy from the embedding of that enemy
    action_dim: int
    hidden_dim: int
    init_scale_emb: float
    init_scale_transf: float
    init_scale_q: float
    transf_num_layers: int
    transf_num_heads: int
    transf_dim_feedforward: int
    transf_dropout_prob: float
    deterministic: bool = True
    num_movement_actions: int = 5
    use_fast_attention: bool = False
    scale_inputs: bool = True
    relu_emb: bool = True

    @nn.compact
    def __call__(self, hs, x, train=True, return_all_hs=False):
        
        ins, resets = x
        # mask for the death/invisible agents, which are assumed to have obs==0
        mask = jnp.all(ins==0, axis=-1).astype(bool) 
        mask = jnp.concatenate((jnp.zeros((*mask.shape[:-1], 1)),mask), axis=-1) # add a positive mask for the agent internal hidden state that will be added later
        embeddings = Embedder(
            self.hidden_dim,
            init_scale=self.init_scale_emb,
            scale_inputs=self.scale_inputs,
            activation=self.relu_emb,
        )(ins, train)
        last_hs, embeddings = ScannedTransformer(
            hidden_dim=self.hidden_dim,
            init_scale=self.init_scale_transf,
            transf_num_layers=self.transf_num_layers,
            transf_num_heads=self.transf_num_heads,
            transf_dim_feedforward=self.transf_dim_feedforward,
            use_fast_attention=self.use_fast_attention,
            deterministic=True,
            return_embeddings=True,
        )(hs, (embeddings, mask, resets))
        
        # q_vals for the movement actions are computed from agents hidden states
        hidden_states = embeddings[..., 0:1, :]
        q_mov = nn.Dense(
            self.num_movement_actions,
            kernel_init=orthogonal(self.init_scale_q),
            bias_init=constant(0.0),
        )(hidden_states) # time_step, batch_size, 1, 5
        
        # q_vals for attacking an enemy is computed from attacking that enemy
        n_enemies = self.action_dim-self.num_movement_actions
        enemy_embeddings = embeddings[..., -n_enemies-1:-1, :] # last embedding is 'self', just before are the enemies

        q_attack = nn.Dense(
            1,
            kernel_init=orthogonal(self.init_scale_q),
            bias_init=constant(0.0)
        )(enemy_embeddings) # time_step, batch_size, n_enemies, 1
        q_vals = jnp.concatenate((q_mov,jnp.swapaxes(q_attack, -1, -2)), axis=-1)
        
        if return_all_hs:
            return last_hs, (hidden_states, q_vals)
        else:
            return last_hs, q_vals
    

class TransformerMixer(nn.Module):

    hidden_dim: int
    init_scale: float
    transf_num_layers: int
    transf_num_heads: int
    transf_dim_feedforward: int
    scale_inputs: bool = True
    use_fast_attention: bool = True
    relu_emb: bool = True
    
    @nn.compact
    def __call__(self, q_vals, hs_agents, states, done, train=True):
        
        n_agents, time_steps, batch_size = q_vals.shape
        q_vals = jnp.transpose(q_vals, (1, 2, 0)) # (time_steps, batch_size, n_agents)

        # the embeddings consist in the state-matrix embeddings and the hidden state of the agents
        hs_agents = hs_agents.reshape(time_steps, batch_size, n_agents, self.hidden_dim)
        mixer_embs = Embedder(
            self.hidden_dim,
            init_scale=self.init_scale,
            scale_inputs=self.scale_inputs,
            activation=self.relu_emb,
        )(states, train)
        mixer_embs = jnp.concatenate((
            mixer_embs,
            hs_agents,
        ), axis=-2)
        hs_mixer = ScannedTransformer.initialize_carry(self.hidden_dim, batch_size, 1)
        _, hyp_emb = ScannedTransformer(
            hidden_dim=self.hidden_dim,
            init_scale=self.init_scale,
            transf_num_layers=self.transf_num_layers,
            transf_num_heads=self.transf_num_heads,
            transf_dim_feedforward=self.transf_dim_feedforward,
            deterministic=True,
            return_embeddings=True,
            use_fast_attention=self.use_fast_attention,
        )(hs_mixer, (mixer_embs, None, done)) # for now the mixer doesn't mask the embeddings
        
        # monotonicity and reshaping
        main_emb = hyp_emb[..., 0:1, :] # main embedding is the hs of the mixer
        w_1 = jnp.abs(hyp_emb[..., -n_agents:, :].reshape(time_steps, batch_size, n_agents, self.hidden_dim)) # w1 is a transformation of the agents' hs
        b_1 = main_emb.reshape(time_steps, batch_size, 1, self.hidden_dim)
        w_2 = jnp.abs(main_emb.reshape(time_steps, batch_size, self.hidden_dim, 1))
        b_2 = nn.Dense(1, kernel_init=orthogonal(self.init_scale), bias_init=constant(0.))(nn.relu(main_emb))
        b_2 = b_2.reshape(time_steps, batch_size, 1, 1)
    
        # mix
        hidden = nn.elu(jnp.matmul(q_vals[:, :, None, :], w_1) + b_1)
        q_tot  = jnp.matmul(hidden, w_2) + b_2
        
        return q_tot.squeeze() # (time_steps, batch_size)
    

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
            chosen_actions = jnp.where(pick_random, random_actions, greedy_actions)
            return chosen_actions
        
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


def tree_mean(tree):
    return jnp.array(
        jax.tree_util.tree_leaves(jax.tree_map(lambda x: x.mean(), tree))
    ).mean()


def make_train(config, env):

    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    def train(rng):

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        wrapped_env = TransformersCTRolloutManager(env, batch_size=config["NUM_ENVS"])
        test_env = TransformersCTRolloutManager(env, batch_size=config["NUM_TEST_EPISODES"]) # batched env for testing (has different batch size)
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
        if 'smax' in env.name.lower(): # smax agent 
            agent_class = TransformerAgentSmax
            n_entities = wrapped_env._env.num_allies+wrapped_env._env.num_enemies # must be explicit for the n_entities if using policy decoupling
            init_x = (
                jnp.zeros((1, 1, n_entities, sample_traj.obs[env.agents[0]].shape[-1])), # (time_step, batch_size, n_entities, obs_size)
                jnp.zeros((1, 1)) # (time_step, batch size)
            )
        else:
            agent_class = TransformerAgent
            init_x = (
                jnp.zeros((1, 1, 1, sample_traj.obs[env.agents[0]].shape[-1])), # (time_step, batch_size, n_entities, obs_size)
                jnp.zeros((1, 1)) # (time_step, batch size)
            )
        agent = agent_class(
            action_dim=wrapped_env.max_action_space,
            hidden_dim=config['AGENT_HIDDEN_DIM'],
            init_scale_emb=config['AGENT_INIT_SCALE'],
            init_scale_transf=config['AGENT_INIT_SCALE'],
            init_scale_q=config['AGENT_INIT_SCALE'],
            transf_num_layers=config['AGENT_TRANSF_NUM_LAYERS'],
            transf_num_heads=config['AGENT_TRANSF_NUM_HEADS'],
            transf_dim_feedforward=config['AGENT_TRANSF_DIM_FF'],
            use_fast_attention=config['USE_FAST_ATTENTION'],
            scale_inputs=config['SCALE_INPUTS'],
            relu_emb=config['EMBEDDER_USE_RELU'],
            transf_dropout_prob=0.,
            deterministic=True,
        )
        rng, _rng = jax.random.split(rng)
        init_hs = ScannedTransformer.initialize_carry(config['AGENT_HIDDEN_DIM'], 1, 1) # (batch_size, hidden_dim)
        agent_params = agent.init(_rng, init_hs, init_x, train=False)

        # init mixer
        rng, _rng = jax.random.split(rng)
        state_size = sample_traj.obs['__all__'].shape[-1]  # get the state shape from the buffer
        init_x = (
            jnp.zeros((len(env.agents), 1, 1)), # q_vals: n_agents, time_steps, batch_size
            ScannedTransformer.initialize_carry(config['AGENT_HIDDEN_DIM'], len(env.agents), 1), # hs_agents: time_step, n_agents*batch_size, hidden_dim
            jnp.zeros((1, 1, 1, state_size)), # states: time_step, batch_size, n_entities, state_size
            jnp.zeros((1, 1)), # done: (time_step, batch size)
            False, # train
        )
        mixer = TransformerMixer(
            hidden_dim=config['AGENT_HIDDEN_DIM'],
            init_scale=config['MIXER_INIT_SCALE'],
            transf_num_layers=config['MIXER_TRANSF_NUM_LAYERS'],
            transf_num_heads=config['MIXER_TRANSF_NUM_HEADS'],
            transf_dim_feedforward=config['MIXER_TRANSF_DIM_FF'],
            scale_inputs=config['SCALE_INPUTS'],
            relu_emb=config['EMBEDDER_USE_RELU'],
            use_fast_attention=config['USE_FAST_ATTENTION'],
        )
        mixer_params = mixer.init(_rng, *init_x)

        # init optimizer
        network_params = {'agent':agent_params['params'],'mixer':mixer_params['params']}
        network_stats  = {'agent':agent_params['batch_stats'],'mixer':mixer_params['batch_stats']}

        # print number of params
        agent_params = sum(x.size for x in jax.tree_util.tree_leaves(network_params['agent']))
        mixer_params = sum(x.size for x in jax.tree_util.tree_leaves(network_params['mixer']))
        jax.debug.print("Number of agent params: {x}", x=agent_params)
        jax.debug.print("Number of mixer params: {x}", x=mixer_params) 
        
        # INIT TRAIN STATE AND OPTIMIZER
        def linear_schedule(count):
            frac = 1.0 - (count / config["NUM_UPDATES"]*config['N_MINI_UPDATES'])
            return config["LR"] * frac
        def exponential_schedule(count):
            return config["LR"] * (1-config['LR_EXP_DECAY_RATE'])**count
        
        decay_type = config.get('LR_DECAY_TYPE', False)

        if decay_type == 'cos':
            lr = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=config["LR"],
                warmup_steps=config['LR_WARMUP'],
                decay_steps=config["NUM_UPDATES"],
                end_value=0.0
            )
        elif decay_type == 'exp':
            lr = exponential_schedule
        elif 'linear':
            lr = linear_schedule
        else:
            lr = config['LR']

        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=lr, eps=config['EPS_ADAM']),
        )

        # to include the batch normalization stats
        class TrainState_(TrainState):
            batch_stats: Any

        train_state = TrainState_.create(
            apply_fn=agent.apply,
            params=network_params,
            batch_stats=network_stats,
            tx=tx,
        )
        # target network params
        copy_tree = lambda tree: jax.tree_map(lambda x: jnp.copy(x), tree)
        target_network_state = {'params':copy_tree(train_state.params), 'batch_stats':copy_tree(train_state.batch_stats)}

        # INIT EXPLORATION STRATEGY
        explorer = EpsilonGreedy(
            start_e=config["EPSILON_START"],
            end_e=config["EPSILON_FINISH"],
            duration=config["EPSILON_ANNEAL_TIME"]
        )

        def homogeneous_pass(params, batch_stats, hidden_state, obs, dones, return_all_hs=False, train=True):

            # concatenate agents and parallel envs to process them in one batch
            agents, flatten_agents_obs = zip(*obs.items())
            original_shape = flatten_agents_obs[0].shape # assumes obs shape is the same for all agents
            batched_input = (
                jnp.concatenate(flatten_agents_obs, axis=1), # (time_step, n_agents*n_envs, n_entities, obs_size)
                jnp.concatenate([dones[agent] for agent in agents], axis=1), # ensure to not pass other keys (like __all__)
            )
            
            # if train, the outs contain the update of the batch norm
            if train:
                outs, batch_norm_update = agent.apply(
                    {'params':params,'batch_stats':batch_stats},
                    hidden_state,
                    batched_input,
                    return_all_hs=return_all_hs,
                    train=True,
                    mutable=['batch_stats']
                )
            else:
                batch_norm_update = None
                outs = agent.apply(
                    {'params':params,'batch_stats':batch_stats},
                    hidden_state,
                    batched_input,
                    return_all_hs=return_all_hs,
                    train=False
                )

            # if return all hs, the outs contain all the hidden states of the agents per each time-step
            if return_all_hs:
                hidden_state, (h_states, q_vals) = outs
            else:
                hidden_state, q_vals = outs

            q_vals = q_vals.reshape(original_shape[0], len(agents), *original_shape[1:-2], -1) # (time_steps, n_agents, n_envs, action_dim)
            q_vals = {a:q_vals[:,i] for i,a in enumerate(agents)}

            if return_all_hs:
                return batch_norm_update, hidden_state, h_states, q_vals
            else:
                return batch_norm_update, hidden_state, q_vals

        # TRAINING LOOP
        def _update_step(runner_state, unused):

            train_state, target_network_state, env_state, buffer_state, time_state, init_obs, init_dones, test_metrics, rng = runner_state

            # EPISODE STEP
            env_params = train_state.params['agent']
            env_batch_norm = train_state.batch_stats['agent']
            def _env_step(step_state, unused):

                env_state, last_obs, last_dones, hstate, rng, t = step_state

                # prepare rngs for actions and step
                rng, key_a, key_s = jax.random.split(rng, 3)

                # SELECT ACTION
                # add a dummy time_step dimension to the agent input
                obs_   = {a:last_obs[a] for a in env.agents} # ensure to not pass the global state (obs["__all__"]) to the network
                obs_   = jax.tree_map(lambda x: x[np.newaxis, :], obs_)
                dones_ = jax.tree_map(lambda x: x[np.newaxis, :], last_dones)
                # get the q_values from the agent netwoek
                _, hstate, q_vals = homogeneous_pass(env_params, env_batch_norm, hstate, obs_, dones_, train=False)
                # remove the dummy time_step dimension and index qs by the valid actions of each agent 
                valid_q_vals = jax.tree_util.tree_map(lambda q, valid_idx: q.squeeze(0)[..., valid_idx], q_vals, wrapped_env.valid_actions)
                # explore with epsilon greedy_exploration
                actions = explorer.choose_actions(valid_q_vals, t, key_a)

                # STEP ENV
                obs, env_state, rewards, dones, infos = wrapped_env.batch_step(key_s, env_state, actions)
                transition = Transition(last_obs, actions, rewards, dones, infos)

                step_state = (env_state, obs, dones, hstate, rng, t+1)
                return step_state, transition


            # prepare the step state and collect the episode trajectory
            rng, _rng = jax.random.split(rng)
            hstate = ScannedTransformer.initialize_carry(config['AGENT_HIDDEN_DIM'], len(env.agents)*config["NUM_ENVS"], 1) # (n_agents*n_envs, hs_size)

            step_state = (
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
            

            def _network_update(carry, unused):

                train_state, rng = carry

                # sample a batched trajectory from the buffer and set the time step dim in first axis
                rng, _rng = jax.random.split(rng)
                learn_traj = buffer.sample(buffer_state, _rng).experience # (batch_size, 1, max_time_steps, ...)
                learn_traj = jax.tree_map(
                    lambda x: jnp.swapaxes(x[:, 0], 0, 1), # remove the dummy sequence dim (1) and swap batch and temporal dims
                    learn_traj
                ) # (max_time_steps, batch_size, ...)
                init_hs = ScannedTransformer.initialize_carry(config['AGENT_HIDDEN_DIM'], len(env.agents)*config["BUFFER_BATCH_SIZE"], 1) # (n_agents*batch_size, hs_size)

                def _loss_fn(params, init_hs, learn_traj):

                    obs_ = {a:learn_traj.obs[a] for a in env.agents} # ensure to not pass the global state (obs["__all__"]) to the network
                    
                    updates_agent, _, hs_agents, q_vals = homogeneous_pass(
                        params['agent'],
                        train_state.batch_stats['agent'],
                        init_hs,
                        obs_,
                        learn_traj.dones,
                        return_all_hs=True,
                        train=True
                    )
                    _, _, hs_target_agents, target_q_vals = homogeneous_pass(
                        target_network_state['params']['agent'],
                        train_state.batch_stats['agent'],
                        init_hs,
                        obs_,
                        learn_traj.dones,
                        return_all_hs=True,
                        train=False
                    )

                    # stop the gradient from passing with the hidden states between agents and mixer
                    hs_agents = jax.lax.stop_gradient(hs_agents)
                    hs_target_agents = jax.lax.stop_gradient(hs_target_agents)

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

                    # compute q_tot with the mixer network
                    chosen_action_qvals_mix, updates_mixer = mixer.apply(
                        {'params':params['mixer'],'batch_stats':train_state.batch_stats['mixer']}, 
                        jnp.stack(list(chosen_action_qvals.values())),
                        hs_agents[:-1], # hs of agents, avoiding last timestep
                        learn_traj.obs['__all__'][:-1], # avoid last timestep
                        learn_traj.dones['__all__'][:-1], # avoid last timestep
                        train=True,
                        mutable=['batch_stats'],
                    )
                    target_max_qvals_mix = mixer.apply(
                        {'params':target_network_state['params']['mixer'],'batch_stats':train_state.batch_stats['mixer']}, 
                        jnp.stack(list(target_max_qvals.values())),
                        hs_target_agents[1:], # hs of target agents, avoiding first timestep
                        learn_traj.obs['__all__'][1:], # avoid first timestep
                        learn_traj.dones['__all__'][1:], # avoid last timestep
                        train=False,
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
                    
                    batch_norm_update = {'agent':updates_agent['batch_stats'], 'mixer':updates_mixer['batch_stats']}
                    return loss, (targets, batch_norm_update)

                # compute loss and optimize grad
                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                (loss, (targets, batch_norm_update)), grads = grad_fn(train_state.params, init_hs, learn_traj)
                train_state = train_state.apply_gradients(grads=grads)
                train_state = train_state.replace(batch_stats=batch_norm_update)

                update_info = {'loss':loss, 'targets':targets.mean(), 'grad':tree_mean(grads)}

                return (train_state, rng), update_info

            # perform n updates over the network
            rng, _rng = jax.random.split(rng)
            update_info_zero = dict(zip(['loss', 'targets', 'grad'], [jnp.zeros(config['N_MINI_UPDATES'])]*3)) # default update info when cannot sample
            (train_state, rng), update_info = jax.lax.cond(
                buffer.can_sample(buffer_state),
                lambda train_state, rng: jax.lax.scan(_network_update, (train_state, rng), None, config['N_MINI_UPDATES']),
                lambda train_state, rng: ((train_state, rng), update_info_zero), # do nothing
                train_state,
                _rng
            )

            # UPDATE THE VARIABLES AND RETURN
            # reset the environment
            rng, _rng = jax.random.split(rng)
            init_obs, env_state = wrapped_env.batch_reset(_rng)
            init_dones = {agent:jnp.zeros((config["NUM_ENVS"]), dtype=bool) for agent in env.agents+['__all__']}

            # update the states
            time_state['timesteps'] = step_state[-1]
            time_state['updates']   = time_state['updates'] + 1

            # update the target network if necessary
            target_network_state = jax.lax.cond(
                time_state['updates'] % config['TARGET_UPDATE_INTERVAL'] == 0,
                lambda _: {'params':copy_tree(train_state.params), 'batch_stats':copy_tree(train_state.batch_stats)},
                lambda _: target_network_state,
                operand=None
            )

            # update the greedy rewards
            rng, _rng = jax.random.split(rng)
            test_metrics = jax.lax.cond(
                time_state['updates'] % (config["TEST_INTERVAL"] // config["NUM_STEPS"] // config["NUM_ENVS"]) == 0,
                lambda _: get_greedy_metrics(_rng, train_state, time_state),
                lambda _: test_metrics,
                operand=None
            )

            # update the returning metrics
            metrics = {
                'running_metrics':{
                    'timesteps': time_state['timesteps']*config['NUM_ENVS'],
                    'updates' : time_state['updates'],
                    'loss': update_info['loss'].mean(),
                    'returns': traj_batch.rewards['__all__'].sum(axis=0).mean(), # mean of sum accross timesteps
                    'targets_mean': update_info['targets'].mean(),
                    'grad_mean':update_info['grad'].mean(),
                    'params_agent_mean':tree_mean(train_state.params['agent']),
                    'params_mixer_mean':tree_mean(train_state.params['mixer']),
                }, 
                'test_metrics': test_metrics
            }

            if config.get('WANDB_ONLINE_REPORT', False):
                def callback(metrics, infos):
                    info_metrics = {
                        k:v[...,0][infos["returned_episode"][..., 0]].mean()
                        for k,v in infos.items() if k!="returned_episode"
                    }
                    wandb.log(
                        {
                            **metrics['running_metrics'],
                            **info_metrics,
                            **{k:v.mean() for k, v in metrics['test_metrics'].items()}
                        }
                    )
                jax.debug.callback(callback, metrics, traj_batch.infos)

            runner_state = (
                train_state,
                target_network_state,
                env_state,
                buffer_state,
                time_state,
                init_obs,
                init_dones,
                test_metrics,
                rng
            )

            if config.get('WANDB_ONLINE_REPORT', False):
                return runner_state, None # don't return metrics if you're using wandb to save memory
            else:
                return runner_state, metrics

        def get_greedy_metrics(rng, train_state, time_state):
            """Help function to test greedy policy during training"""
            env_params = train_state.params['agent']
            env_batch_norm = train_state.batch_stats['agent']
            def _greedy_env_step(step_state, unused):
                env_state, last_obs, last_dones, hstate, rng = step_state
                rng, key_s = jax.random.split(rng)
                obs_   = {a:last_obs[a] for a in env.agents}
                obs_   = jax.tree_map(lambda x: x[np.newaxis, :], obs_)
                dones_ = jax.tree_map(lambda x: x[np.newaxis, :], last_dones)
                _, hstate, q_vals = homogeneous_pass(env_params, env_batch_norm, hstate, obs_, dones_, train=False)
                actions = jax.tree_util.tree_map(lambda q, valid_idx: jnp.argmax(q.squeeze(0)[..., valid_idx], axis=-1), q_vals, test_env.valid_actions)
                obs, env_state, rewards, dones, infos = test_env.batch_step(key_s, env_state, actions)
                step_state = (env_state, obs, dones, hstate, rng)
                return step_state, (rewards, dones, infos)
            rng, _rng = jax.random.split(rng)
            init_obs, env_state = test_env.batch_reset(_rng)
            init_dones = {agent:jnp.zeros((config["NUM_TEST_EPISODES"]), dtype=bool) for agent in env.agents+['__all__']}
            rng, _rng = jax.random.split(rng)
            hstate = ScannedTransformer.initialize_carry(config['AGENT_HIDDEN_DIM'], len(env.agents)*config["NUM_TEST_EPISODES"], 1) # (n_agents*n_envs, hs_size)
            step_state = (
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
        test_metrics = get_greedy_metrics(_rng, train_state, time_state) # initial greedy metrics
        
        # train
        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            target_network_state,
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


def single_run(config):
    """Perform a single run with multiple parallel seeds in one env."""
    config = OmegaConf.to_container(config)

    print('Config:\n', OmegaConf.to_yaml(config))

    env_name = config["env"]["ENV_NAME"]
    alg_name = f'transf_qmix'
    
    # smax init neeeds a scenario
    if 'smax' in env_name.lower():
        config['env']['ENV_KWARGS']['scenario'] = map_name_to_scenario(config['env']['MAP_NAME'])
        env_name = f"{config['env']['ENV_NAME']}_{config['env']['MAP_NAME']}"
        env = make(config["env"]["ENV_NAME"], **config['env']['ENV_KWARGS'])
        env = SMAXLogWrapper(env)
    # overcooked needs a layout 
    elif 'overcooked' in env_name.lower():
        config['env']["ENV_KWARGS"]["layout"] = overcooked_layouts[config['env']["ENV_KWARGS"]["layout"]]
        env = make(config["env"]["ENV_NAME"], **config['env']['ENV_KWARGS'])
        env = LogWrapper(env)
    else:
        env = make(config["env"]["ENV_NAME"], **config['env']['ENV_KWARGS'])
        env = LogWrapper(env)

    #config["alg"]["NUM_STEPS"] = config["alg"].get("NUM_STEPS", env.max_steps) # default steps defined by the env
    
    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=[
            alg_name.upper(),
            env_name.upper(),
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


def tune(default_config):
    """Hyperparameter sweep with wandb."""
    import copy

    default_config = OmegaConf.to_container(default_config)

    print('Config:\n', OmegaConf.to_yaml(default_config))

    def wrapped_make_train():

        wandb.init(project=default_config['PROJECT'])
        # update the default params
        config = copy.deepcopy(default_config)
        for k, v in dict(wandb.config).items():
            config['alg'][k] = v
            
        print('running experiment with params:', config)
        
        env_name = config["env"]["ENV_NAME"]
        # smac init neeeds a scenario
        if 'smax' in env_name.lower():
            config['env']['ENV_KWARGS']['scenario'] = map_name_to_scenario(config['env']['MAP_NAME'])
            env_name = f"{config['env']['ENV_NAME']}_{config['env']['MAP_NAME']}"
            env = make(config["env"]["ENV_NAME"], **config['env']['ENV_KWARGS'])
            env = SMAXLogWrapper(env)
        # overcooked needs a layout 
        elif 'overcooked' in env_name.lower():
            config['env']["ENV_KWARGS"]["layout"] = overcooked_layouts[config['env']["ENV_KWARGS"]["layout"]]
            env = make(config["env"]["ENV_NAME"], **config['env']['ENV_KWARGS'])
            env = LogWrapper(env)
        else:
            env = make(config["env"]["ENV_NAME"], **config['env']['ENV_KWARGS'])
            env = LogWrapper(env)

        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, config["NUM_SEEDS"])
        train_vjit = jax.jit(jax.vmap(make_train(config["alg"], env)))
        outs = jax.block_until_ready(train_vjit(rngs))

    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'test_returns',
            'goal': 'maximize',
        },
        'parameters':{
            'LR':{'values':[0.005, 0.001, 0.0005]},
            'EPS_ADAM':{'values':[0.0001, 0.0000001, 0.0000000001]},
            'SCALE_INPUTS':{'values':[True, False]},
            'NUM_ENVS':{'values':[8, 16]},
            'N_MINI_UPDATES':{'values':[1, 2, 4]},
        },
    }

    wandb.login()
    sweep_id = wandb.sweep(sweep_config, entity=default_config['ENTITY'],project=default_config['PROJECT'])
    wandb.agent(sweep_id, wrapped_make_train, count=100)

@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(config):
    #tune(config) # uncomment to run hypertuning
    single_run(config)

if __name__ == "__main__":
    main()