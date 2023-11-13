import jax
from jax import numpy as jnp
import numpy as np
import jax.experimental.checkify as checkify
import chex
import flax.linen as nn
from gymnax.environments.spaces import Box as BoxGymnax, Discrete as DiscreteGymnax
from jaxmarl.environments.spaces import Box, Discrete, MultiDiscrete
from jaxmarl.environments.multi_agent_env import MultiAgentEnv

from .buffers import uniform_replay
from .buffers.uniform import UniformReplayBufferState

from typing import NamedTuple, List
from functools import partial

from safetensors.flax import save_file, load_file
from flax.traverse_util import flatten_dict, unflatten_dict
import typing
import os 

def save_params(params: typing.Dict, filename: typing.Union[str, os.PathLike]) -> None:
    flattened_dict = flatten_dict(params, sep=',')
    save_file(flattened_dict, filename)

def load_params(filename: typing.Union[str, os.PathLike]) -> typing.Dict:
    flattened_dict = load_file(filename)
    return unflatten_dict(flattened_dict, sep=',')


def get_space_dim(space):
    if isinstance(space, (DiscreteGymnax, Discrete)):
        return space.n
    elif isinstance(space, (BoxGymnax, Box, MultiDiscrete)):
        return np.prod(space.shape)
    else:
        print(space)
        raise NotImplementedError('Current wrapper works only with Discrete/MultiDiscrete/Box action and obs spaces')


class CTRolloutManager:
    """
    Rollout Manager for Centralized Training with Parameters Sharing.
    - Batchify multiple environments (batch_size in __init__).
    - Adds a global state (obs["__all__"]) and a global reward (rewards["__all__"]) in the env.step returns.
    - Pads the observations of the agents in order to have all the same length.
    - Adds an agent id (one hot encoded) to the observation vectors.

    By default:
    - global_state is the concatenation of all agents' observations.
    - global_reward is the sum of all agents' rewards.
    """

    def __init__(self, env: MultiAgentEnv, batch_size:int, training_agents:List=None):
        self.env = env
        self.batch_size = batch_size

        self.agents = env.agents

        # the agents to train could differ from the total trainable agents in the env (f.i. if using pretrained agents)
        # it's important to know it in order to compute properly the default global rewards and state
        self.training_agents = self.agents if training_agents is None else training_agents    

        # TOREMOVE: this is because overcooked doesn't follow other envs conventions
        if len(env.observation_spaces) == 0:
            env.observation_spaces = {agent:env.observation_space() for agent in self.agents}
        if len(env.action_spaces) == 0:
            env.action_spaces = {agent:env.action_space() for agent in self.agents}
        
        # batched action sampling
        self.batch_samplers = {agent: jax.jit(jax.vmap(env.action_space(agent).sample, in_axes=0)) for agent in self.agents}

        # assumes the observations are flattened vectors
        self.max_obs_length = max(list(map(lambda x: get_space_dim(x), env.observation_spaces.values())))
        self.max_action_space = max(list(map(lambda x: get_space_dim(x), env.action_spaces.values())))
        self.obs_size = self.max_obs_length + len(self.agents)

        # agents ids
        self.agents_one_hot = {a:oh for a, oh in zip(self.agents, jnp.eye(len(self.agents)))}
        # valid actions
        self.valid_actions = {a:jnp.arange(u.n) for a, u in env.action_spaces.items()}
        self.valid_actions_oh ={a:jnp.concatenate((jnp.ones(u.n), jnp.zeros(self.max_action_space - u.n))) for a, u in env.action_spaces.items()}

        # custom global state and rewards for specific envs
        if 'smac' in env.name.lower():
            self.global_state = lambda obs, state: obs['world_state']
            self.global_reward = lambda rewards: rewards[self.training_agents[0]]
    
    @partial(jax.jit, static_argnums=0)
    def batch_reset(self, key):
        keys = jax.random.split(key, self.batch_size)
        return jax.vmap(self.wrapped_reset, in_axes=0)(keys)

    @partial(jax.jit, static_argnums=0)
    def batch_step(self, key, states, actions):
        keys = jax.random.split(key, self.batch_size)
        return jax.vmap(self.wrapped_step, in_axes=(0, 0, 0))(keys, states, actions)

    @partial(jax.jit, static_argnums=0)
    def wrapped_reset(self, key):
        obs_, state = self.env.reset(key)
        obs = jax.tree_util.tree_map(self._preprocess_obs, {agent:obs_[agent] for agent in self.agents}, self.agents_one_hot)
        obs["__all__"] = self.global_state(obs_, state)
        return obs, state

    @partial(jax.jit, static_argnums=0)
    def wrapped_step(self, key, state, actions):
        obs_, state, reward, done, infos = self.env.step(key, state, actions)
        obs = jax.tree_util.tree_map(self._preprocess_obs, {agent:obs_[agent] for agent in self.agents}, self.agents_one_hot)
        obs = jax.tree_util.tree_map(lambda d, o: jnp.where(d, 0., o), {agent:done[agent] for agent in self.agents}, obs) # ensure that the obs are 0s for done agents
        obs["__all__"] = self.global_state(obs_, state)
        reward["__all__"] = self.global_reward(reward)
        return obs, state, reward, done, infos

    @partial(jax.jit, static_argnums=0)
    def global_state(self, obs, state):
        return jnp.concatenate([obs[agent] for agent in self.agents], axis=-1)
    
    @partial(jax.jit, static_argnums=0)
    def global_reward(self, reward):
        return jnp.stack([reward[agent] for agent in self.training_agents]).sum(axis=0) 
    
    def batch_sample(self, key, agent):
        return self.batch_samplers[agent](jax.random.split(key, self.batch_size))

    @partial(jax.jit, static_argnums=0)
    def _preprocess_obs(self, arr, extra_features):
        # flatten
        arr = arr.flatten()
        # pad the observation vectors to the maximum length
        pad_width = [(0, 0)] * (arr.ndim - 1) + [(0, max(0, self.max_obs_length - arr.shape[-1]))]
        arr = jnp.pad(arr, pad_width, mode='constant', constant_values=0)
        # concatenate the extra features
        arr = jnp.concatenate((arr, extra_features), axis=-1)
        return arr


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
        choosed_actions = jax.tree_map(lambda q, k: explore(q, eps, k), q_vals, keys)
        return choosed_actions
    

class Transition(NamedTuple):
    obs: dict
    actions: dict
    rewards: dict
    dones: dict

class UniformBuffer:
    # Uniform Buffer replay buffer aggregating transitions from parallel envs
    # based on dejax: https://github.com/hr0nix/dejax/tree/main
    def __init__(self, parallel_envs:int=10, batch_size:int=32, max_size:int=5000):
        self.batch_size = batch_size
        self.buffer = uniform_replay(max_size=max_size)
        self.parallel_envs = parallel_envs
        self.sample = checkify.checkify(self.sample)
        
    def reset(self, transition_sample: Transition) -> UniformReplayBufferState:
        zero_transition = jax.tree_util.tree_map(jnp.zeros_like, transition_sample)
        return self.buffer.init_fn(zero_transition)
    
    @partial(jax.jit, static_argnums=0)
    def add(self, buffer_state: UniformReplayBufferState, transition: Transition) -> UniformReplayBufferState:
        def add_to_buffer(i, buffer_state):
            # assumes the transition is coming from jax.lax so the batch is on dimension 1
            return self.buffer.add_fn(buffer_state, jax.tree_util.tree_map(lambda x: x[:, i], transition))
        # need to use for and not vmap because you can't add multiple transitions on the same buffer in parallel
        return jax.lax.fori_loop(0, self.parallel_envs, add_to_buffer, buffer_state)
    
    @partial(jax.jit, static_argnums=0)
    def sample(self, buffer_state: UniformReplayBufferState, key: chex.PRNGKey) -> Transition:
        return self.buffer.sample_fn(buffer_state, key, self.batch_size)


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