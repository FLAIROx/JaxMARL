import jax
from jax import numpy as jnp
import numpy as np
import jax.experimental.checkify as checkify
import chex
from gymnax.environments.spaces import Box as BoxGymnax, Discrete as DiscreteGymnax
from smax.environments.spaces import Box, Discrete, MultiDiscrete
from smax.environments.multi_agent_env import MultiAgentEnv

from .buffers import uniform_replay
from .buffers.uniform import UniformReplayBufferState

from typing import NamedTuple
from functools import partial

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

    def __init__(self, env: MultiAgentEnv, batch_size:int):
        self.env = env
        self.batch_size = batch_size

        # TOREMOVE: this is because overcooked doesn't follow other envs conventions
        if len(env.observation_spaces) == 0:
            env.observation_spaces = {agent:env.observation_space() for agent in env.agents}
        if len(env.action_spaces) == 0:
            env.action_spaces = {agent:env.action_space() for agent in env.agents}
        
        # batched action sampling
        self.batch_samplers = {agent: jax.jit(jax.vmap(env.action_space(agent).sample, in_axes=0)) for agent in self.env.agents}

        # assumes the observations are flattened vectors
        self.max_obs_length = max(list(map(lambda x: get_space_dim(x), env.observation_spaces.values())))
        self.max_action_space = max(list(map(lambda x: get_space_dim(x), env.action_spaces.values())))
        self.obs_size = self.max_obs_length + len(env.agents)

        # agents ids
        self.agents_one_hot = {a:oh for a, oh in zip(env.agents, jnp.eye(len(env.agents)))}
        # valid actions
        self.valid_actions = {a:jnp.arange(u.n) for a, u in env.action_spaces.items()}

        # custom global state for specific envs
        if 'smac' in env.name.lower():
            self.global_state = lambda obs, state: self.env._env.get_world_state(state)
    
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
        obs, state = self.env.reset(key)
        obs = jax.tree_util.tree_map(self._preprocess_obs, obs, self.agents_one_hot)
        obs["__all__"] = self.global_state(obs, state)
        return obs, state

    @partial(jax.jit, static_argnums=0)
    def wrapped_step(self, key, state, actions):
        obs, state, reward, done, infos = self.env.step(key, state, actions)
        obs = jax.tree_util.tree_map(self._preprocess_obs, obs, self.agents_one_hot)
        obs["__all__"] = self.global_state(obs, state)
        reward["__all__"] = self.global_reward(reward)
        return obs, state, reward, done, infos

    @partial(jax.jit, static_argnums=0)
    def global_state(self, obs, state):
        return jnp.concatenate(list(obs.values()), axis=-1)
    
    @partial(jax.jit, static_argnums=0)
    def global_reward(self, reward):
        return jnp.stack(list(reward.values())).sum(axis=0) 
    
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