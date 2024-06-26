import os
import copy
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import Any

import chex
import optax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from gymnax.wrappers.purerl import LogWrapper
import hydra
from omegaconf import OmegaConf
import flashbax as fbx
import wandb

from jaxmarl import make
from jaxmarl.environments.smax import map_name_to_scenario
from jaxmarl.environments.overcooked import overcooked_layouts
from jaxmarl.wrappers.baselines import (
    SMAXLogWrapper,
    MPELogWrapper,
    LogWrapper,
    CTRolloutManager,
)


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
    def __call__(self, hs, x, train=False, return_all_hs=False):
        
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


@chex.dataclass(frozen=True)
class Timestep:
    obs: dict
    actions: dict
    rewards: dict
    dones: dict
    avail_actions: dict


class CustomTrainState(TrainState):
    target_network_params: Any
    batch_stats: Any
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0

def make_train(config, env):

    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    eps_scheduler = optax.linear_schedule(
        init_value=config["EPS_START"],
        end_value=config["EPS_FINISH"],
        transition_steps=config["EPS_DECAY"] * config["NUM_UPDATES"],
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
        original_seed = rng[0]
        rng, _rng = jax.random.split(rng)
        wrapped_env = CTRolloutManager(env, batch_size=config["NUM_ENVS"])
        test_env = CTRolloutManager(
            env, batch_size=config["TEST_NUM_ENVS"]
        )  # batched env for testing (has different batch size)

        # to initalize some variables is necessary to sample a trajectory to know its strucutre
        def _env_sample_step(env_state, unused):
            rng, key_a, key_s = jax.random.split(
                jax.random.PRNGKey(0), 3
            )  # use a dummy rng here
            key_a = jax.random.split(key_a, env.num_agents)
            actions = {
                agent: wrapped_env.batch_sample(key_a[i], agent)
                for i, agent in enumerate(env.agents)
            }
            avail_actions = wrapped_env.get_valid_actions(env_state.env_state)
            obs, env_state, rewards, dones, infos = wrapped_env.batch_step(
                key_s, env_state, actions
            )
            timestep = Timestep(
                obs=obs,
                actions=actions,
                rewards=rewards,
                dones=dones,
                avail_actions=avail_actions,
            )
            return env_state, timestep

        _, _env_state = wrapped_env.batch_reset(rng)
        _, sample_traj = jax.lax.scan(
            _env_sample_step, _env_state, None, config["NUM_STEPS"]
        )
        sample_traj_unbatched = jax.tree_map(
            lambda x: x[:, 0], sample_traj
        )  # remove the NUM_ENV dim

        # init agent
        agent_class = TransformerAgentSmax if 'smax' in env.name.lower() else TransformerAgent
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

        def create_agent(rng):
            if 'smax' in env.name.lower(): # smax agent 
                n_entities = wrapped_env._env.num_allies+wrapped_env._env.num_enemies # must be explicit for the n_entities if using policy decoupling
                init_x = (
                    jnp.zeros((1, 1, n_entities, sample_traj.obs[env.agents[0]].shape[-1])), # (time_step, batch_size, n_entities, obs_size)
                    jnp.zeros((1, 1)) # (time_step, batch size)
                )
            else:
                init_x = (
                    jnp.zeros((1, 1, 1, sample_traj.obs[env.agents[0]].shape[-1])), # (time_step, batch_size, n_entities, obs_size)
                    jnp.zeros((1, 1)) # (time_step, batch size)
                )
            rng, _rng = jax.random.split(rng)
            init_hs = ScannedTransformer.initialize_carry(config['AGENT_HIDDEN_DIM'], 1, 1) # (batch_size, hidden_dim)
            agent_params = agent.init(_rng, init_hs, init_x, train=False)

            # init mixer
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
            mixer_params = mixer.init(_rng, *init_x)

            network_params = {'agent':agent_params['params'],'mixer':mixer_params['params']}
            network_stats  = {'agent':agent_params['batch_stats'],'mixer':mixer_params['batch_stats']}

            def exponential_schedule(count):
                return config["LR"] * (1-config['LR_EXP_DECAY_RATE'])**count

            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.radam(learning_rate=exponential_schedule),
            )

            train_state = CustomTrainState.create(
                apply_fn=agent.apply,
                params=network_params,
                batch_stats=network_stats,
                target_network_params=network_params,
                tx=tx,
            )
            return train_state

        rng, _rng = jax.random.split(rng)
        train_state = create_agent(rng)

        # INIT BUFFER
        buffer = fbx.make_trajectory_buffer(
            max_length_time_axis=config["BUFFER_SIZE"] // config["NUM_ENVS"],
            min_length_time_axis=config["BUFFER_BATCH_SIZE"],
            sample_batch_size=config["BUFFER_BATCH_SIZE"],
            add_batch_size=config["NUM_ENVS"],
            sample_sequence_length=1,
            period=1,
        )
        buffer_state = buffer.init(sample_traj_unbatched)

        # TRAINING LOOP
        def _update_step(runner_state, unused):

            train_state, buffer_state, test_state, rng = runner_state

            # SAMPLE PHASE
            def _step_env(carry, _):
                hs, last_obs, last_dones, env_state, rng = carry
                rng, rng_a, rng_s = jax.random.split(rng, 3)

                # (num_agents, 1 (dummy time), num_envs, obs_size)
                _obs = batchify(last_obs)[:, np.newaxis]
                _dones = batchify(last_dones)[:, np.newaxis]

                new_hs, q_vals = jax.vmap(agent.apply, in_axes=(None, 0, 0, 0, None))(
                    {
                        "params": train_state.params['agent'],
                        "batch_stats": train_state.batch_stats['agent'],
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
                actions = jax.vmap(eps_greedy_exploration, in_axes=(0, 0, None, 0))(
                    _rngs, q_vals, eps, batchify(avail_actions)
                )
                actions = unbatchify(actions)

                new_obs, new_env_state, rewards, dones, infos = wrapped_env.batch_step(
                    rng_s, env_state, actions
                )
                timestep = Timestep(
                    obs=last_obs,
                    actions=actions,
                    rewards=jax.tree_map(lambda x:config.get("REW_SCALE", 1)*x, rewards),
                    dones=dones,
                    avail_actions=avail_actions,
                )
                return (new_hs, new_obs, dones, new_env_state, rng), (timestep, infos)

            # step the env (should be a complete rollout)
            rng, _rng = jax.random.split(rng)
            init_obs, env_state = wrapped_env.batch_reset(_rng)
            init_dones = {
                agent: jnp.zeros((config["NUM_ENVS"]), dtype=bool)
                for agent in env.agents + ["__all__"]
            }
            init_hs = ScannedTransformer.initialize_carry(
                config["HIDDEN_SIZE"], len(env.agents), config["NUM_ENVS"]
            )
            expl_state = (init_hs, init_obs, init_dones, env_state)
            rng, _rng = jax.random.split(rng)
            _, (timesteps, infos) = jax.lax.scan(
                _step_env,
                (*expl_state, _rng),
                None,
                config["NUM_STEPS"],
            )

            train_state = train_state.replace(
                timesteps=train_state.timesteps
                + config["NUM_STEPS"] * config["NUM_ENVS"]
            )  # update timesteps count

            # BUFFER UPDATE
            buffer_traj_batch = jax.tree_util.tree_map(
                lambda x: jnp.swapaxes(x, 0, 1)[
                    :, np.newaxis
                ],  # put the batch dim first and add a dummy sequence dim
                timesteps,
            )  # (num_envs, 1, time_steps, ...)
            buffer_state = buffer.add(buffer_state, buffer_traj_batch)

            # NETWORKS UPDATE
            def _learn_phase(carry, _):

                train_state, rng = carry
                rng, _rng = jax.random.split(rng)
                minibatch = buffer.sample(buffer_state, _rng).experience
                minibatch = jax.tree_map(
                    lambda x: jnp.swapaxes(
                        x[:, 0], 0, 1
                    ),  # remove the dummy sequence dim (1) and swap batch and temporal dims
                    minibatch,
                )  # (max_time_steps, batch_size, ...)

                # preprocess network input
                init_hs = ScannedTransformer.initialize_carry(
                    config["HIDDEN_SIZE"],
                    len(env.agents),
                    config["BUFFER_BATCH_SIZE"],
                )
                # num_agents, timesteps, batch_size, ...
                _obs = batchify(minibatch.obs)
                _dones = batchify(minibatch.dones)
                _actions = batchify(minibatch.actions)
                # _rewards = batchify(minibatch.rewards)
                _avail_actions = batchify(minibatch.avail_actions)

                _, (h_states, q_next_target) = jax.vmap(agent.apply, in_axes=(None, 0, 0, 0, None, None))(
                   {"params": train_state.target_network_params['agent'], "batch_stats": train_state.batch_stats['agent']},
                    init_hs,
                    _obs,
                    _dones,
                    False,
                    True
                )  # (num_agents, timesteps, batch_size, num_actions)

                def _loss_fn(params):
                    (_, (h_states, q_vals)), updates = jax.vmap(agent.apply, in_axes=(None, 0, 0, 0, None, None))(
                    {"params": params['agent'], "batch_stats": train_state.batch_stats['agent']},
                        init_hs,
                        _obs,
                        _dones,
                        True,
                        True
                    )  # (num_agents, timesteps, batch_size, num_actions)

                    # get logits of the chosen actions
                    chosen_action_q_vals = jnp.take_along_axis(
                        q_vals,
                        _actions[..., np.newaxis],
                        axis=-1,
                    ).squeeze(-1)  # (num_agents, timesteps, batch_size,)

                    unavailable_actions = 1 - _avail_actions
                    valid_q_vals = q_vals - (unavailable_actions * 1e10)

                    # get the q values of the next state
                    q_next = jnp.take_along_axis(
                        q_next_target,
                        jnp.argmax(valid_q_vals, axis=-1)[..., np.newaxis],
                        axis=-1,
                    ).squeeze(-1)  # (num_agents, timesteps, batch_size,)

                    qmix_next = mixer.apply(train_state.target_network_params['mixer'], q_next, minibatch.obs["__all__"])
                    qmix_target = (
                        minibatch.rewards["__all__"][:-1]
                        + (
                            1 - minibatch.dones["__all__"][:-1]
                        )  # use next done because last done was saved for rnn re-init
                        * config["GAMMA"]
                        * qmix_next[1:]  # sum over agents
                    )

                    qmix = mixer.apply(params['mixer'], chosen_action_q_vals, minibatch.obs["__all__"])[:-1]
                    loss = jnp.mean(
                        (qmix - jax.lax.stop_gradient(qmix_target)) ** 2
                    )

                    return loss, chosen_action_q_vals.mean()

                (loss, qvals), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
                    train_state.params
                )
                train_state = train_state.apply_gradients(grads=grads)
                train_state = train_state.replace(
                    grad_steps=train_state.grad_steps + 1,
                )
                return (train_state, rng), (loss, qvals)

            rng, _rng = jax.random.split(rng)
            is_learn_time = (
                buffer.can_sample(buffer_state)
            ) & (  # enough experience in buffer
                train_state.timesteps > config["LEARNING_STARTS"]
            )
            (train_state, rng), (loss, qvals) = jax.lax.cond(
                is_learn_time,
                lambda train_state, rng: jax.lax.scan(
                    _learn_phase, (train_state, rng), None, config["NUM_EPOCHS"]
                ),
                lambda train_state, rng: (
                    (train_state, rng),
                    (
                        jnp.zeros(config["NUM_EPOCHS"]),
                        jnp.zeros(config["NUM_EPOCHS"]),
                    ),
                ),  # do nothing
                train_state,
                _rng,
            )

            # update target network
            train_state = jax.lax.cond(
                train_state.n_updates % config["TARGET_UPDATE_INTERVAL"] == 0,
                lambda train_state: train_state.replace(
                    target_network_params=optax.incremental_update(
                        train_state.params,
                        train_state.target_network_params,
                        config["TAU"],
                    )
                ),
                lambda train_state: train_state,
                operand=train_state,
            )

            # UPDATE METRICS
            train_state = train_state.replace(n_updates=train_state.n_updates + 1)
            metrics = {
                "env_step": train_state.timesteps,
                "update_steps": train_state.n_updates,
                "grad_steps": train_state.grad_steps,
                "loss": loss.mean(),
                "qvals": qvals.mean(),
            }
            metrics.update(jax.tree_map(lambda x: x.mean(), infos))

            # update the test metrics
            if config.get("TEST_DURING_TRAINING", True):
                rng, _rng = jax.random.split(rng)
                test_state = jax.lax.cond(
                    train_state.n_updates
                    % int(config["NUM_UPDATES"] * config["TEST_INTERVAL"])
                    == 0,
                    lambda _: get_greedy_metrics(_rng, train_state),
                    lambda _: test_state,
                    operand=None,
                )
                metrics.update({"test_" + k: v for k, v in test_state.items()})

            # report on wandb if required
            if config["WANDB_MODE"] != "disabled":

                def callback(metrics, original_seed):
                    if config.get('WANDB_LOG_ALL_SEEDS', False):
                        metrics.update(
                            {f"rng{int(original_seed)}/{k}": v for k, v in metrics.items()}
                        )
                    wandb.log(metrics)

                jax.debug.callback(callback, metrics, original_seed)

            runner_state = (train_state, buffer_state, test_state, rng)

            return runner_state, None

        def get_greedy_metrics(rng, train_state):
            """Help function to test greedy policy during training"""
            if not config.get("TEST_DURING_TRAINING", True):
                return None
            
            params = train_state.params['agent']  
            def _greedy_env_step(step_state, unused):
                params, env_state, last_obs, last_dones, hstate, rng = step_state
                rng, key_s = jax.random.split(rng)
                _obs = batchify(last_obs)[:, np.newaxis]
                _dones = batchify(last_dones)[:, np.newaxis]
                hstate, q_vals = jax.vmap(network.apply, in_axes=(None, 0, 0, 0))(
                    params,
                    hstate,
                    _obs,
                    _dones,
                )
                q_vals = q_vals.squeeze(axis=1)
                valid_actions = test_env.get_valid_actions(env_state.env_state)
                actions = get_greedy_actions(q_vals, batchify(valid_actions))
                actions = unbatchify(actions)
                obs, env_state, rewards, dones, infos = test_env.batch_step(
                    key_s, env_state, actions
                )
                step_state = (params, env_state, obs, dones, hstate, rng)
                return step_state, (rewards, dones, infos)

            rng, _rng = jax.random.split(rng)
            init_obs, env_state = test_env.batch_reset(_rng)
            init_dones = {
                agent: jnp.zeros((config["TEST_NUM_ENVS"]), dtype=bool)
                for agent in env.agents + ["__all__"]
            }
            rng, _rng = jax.random.split(rng)
            hstate = ScannedRNN.initialize_carry(
                config["HIDDEN_SIZE"], len(env.agents), config["TEST_NUM_ENVS"]
            )  # (n_agents*n_envs, hs_size)
            step_state = (
                params,
                env_state,
                init_obs,
                init_dones,
                hstate,
                _rng,
            )
            step_state, (rewards, dones, infos) = jax.lax.scan(
                _greedy_env_step, step_state, None, config["TEST_NUM_STEPS"]
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

        # train
        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, buffer_state, test_state, _rng)

        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )

        return {"runner_state": runner_state, "metrics": metrics}

    return train


def env_from_config(config):
    env_name = config["ENV_NAME"]
    # smax init neeeds a scenario
    if "smax" in env_name.lower():
        config["ENV_KWARGS"]["scenario"] = map_name_to_scenario(config["MAP_NAME"])
        env_name = f"{config['ENV_NAME']}_{config['MAP_NAME']}"
        env = make(config["ENV_NAME"], **config["ENV_KWARGS"])
        env = SMAXLogWrapper(env)
    # overcooked needs a layout
    elif "overcooked" in env_name.lower():
        env_name = f"{config['ENV_NAME']}_{config['ENV_KWARGS']['layout']}"
        config["ENV_KWARGS"]["layout"] = overcooked_layouts[
            config["ENV_KWARGS"]["layout"]
        ]
        env = make(config["ENV_NAME"], **config["ENV_KWARGS"])
        env = LogWrapper(env)
    elif "mpe" in env_name.lower():
        env = make(config["ENV_NAME"], **config["ENV_KWARGS"])
        env = MPELogWrapper(env)
    else:
        env = make(config["ENV_NAME"], **config["ENV_KWARGS"])
        env = LogWrapper(env)
    return env, env_name


def single_run(config):

    config = {**config, **config["alg"]}  # merge the alg config with the main config
    print("Config:\n", OmegaConf.to_yaml(config))

    alg_name = config.get("ALG_NAME", "qmix_rnn")
    env, env_name = env_from_config(copy.deepcopy(config))

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

    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_vjit = jax.jit(jax.vmap(make_train(config, env)))
    outs = jax.block_until_ready(train_vjit(rngs))

    # save params
    if config.get("SAVE_PATH", None) is not None:
        from jaxmarl.wrappers.baselines import save_params

        model_state = outs["runner_state"][0]
        save_dir = os.path.join(config["SAVE_PATH"], env_name)
        os.makedirs(save_dir, exist_ok=True)
        OmegaConf.save(
            config,
            os.path.join(
                save_dir, f'{alg_name}_{env_name}_seed{config["SEED"]}_config.yaml'
            ),
        )

        for i, rng in enumerate(rngs):
            params = jax.tree_map(lambda x: x[i], model_state.params)
            save_path = os.path.join(
                save_dir,
                f'{alg_name}_{env_name}_seed{config["SEED"]}_vmap{i}.safetensors',
            )
            save_params(params, save_path)


def tune(default_config):
    """Hyperparameter sweep with wandb."""

    default_config = {**default_config, **default_config["alg"]}  # merge the alg config with the main config
    env_name = default_config["ENV_NAME"]
    alg_name = default_config.get("ALG_NAME", "qmix_rnn")
    env, env_name = env_from_config(default_config)

    def wrapped_make_train():
        wandb.init(project=default_config["PROJECT"])

        # update the default params
        config = copy.deepcopy(default_config)
        for k, v in dict(wandb.config).items():
            config[k] = v

        print("running experiment with params:", config)

        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, config["NUM_SEEDS"])
        train_vjit = jax.jit(jax.vmap(make_train(config, env)))
        outs = jax.block_until_ready(train_vjit(rngs))

    sweep_config = {
        "name": f"{alg_name}_{env_name}",
        "method": "bayes",
        "metric": {
            "name": "test_returned_episode_returns",
            "goal": "maximize",
        },
        "parameters": {
            "LR": {
                "values": [
                    0.005,
                    0.001,
                    0.0005,
                    0.0001,
                    0.00005,
                ]
            },
            "NUM_ENVS": {"values": [8, 32, 64, 128]},
        },
    }

    wandb.login()
    sweep_id = wandb.sweep(
        sweep_config, entity=default_config["ENTITY"], project=default_config["PROJECT"]
    )
    wandb.agent(sweep_id, wrapped_make_train, count=300)


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
