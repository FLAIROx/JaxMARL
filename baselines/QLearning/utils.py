from smax.environments.multi_agent_env import MultiAgentEnv
from functools import partial
import jax
from jax import numpy as jnp

class RolloutManager:
    """Menages parallel environments"""
    def __init__(self, env: MultiAgentEnv, batch_size:int):
        self.env = env
        self.batch_size = batch_size
        self.batch_samplers = {agent: jax.jit(jax.vmap(env.action_space(agent).sample, in_axes=0)) for agent in self.env.agents}

    @partial(jax.jit, static_argnums=0)
    def batch_reset(self, key):
        keys = jax.random.split(key, self.batch_size)
        return jax.vmap(self.env.reset, in_axes=0)(keys)

    @partial(jax.jit, static_argnums=0)
    def batch_step(self, key, states, actions):
        keys = jax.random.split(key, self.batch_size)
        return jax.vmap(self.env.step, in_axes=(0, 0, 0))(keys, states, actions)

    def batch_sample(self, key, agent):
        return self.batch_samplers[agent](jax.random.split(key, self.batch_size))


class CTRolloutManager(RolloutManager):
    """
    Rollout Manager for Centralized Training.
    Adds a global state (obs["__all__"]) and a global reward (rewards["__all__"]) in the env.step returns.

    By default:
    - global_state is the concatenation of all agents' observations.
    - global_reward is the sum of all agents' rewards.
    """

    def __init__(self, env: MultiAgentEnv, batch_size:int):
        super().__init__(env, batch_size)
    
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
        obs["__all__"]    = self.global_state(obs)
        return obs, state

    @partial(jax.jit, static_argnums=0)
    def wrapped_step(self, key, state, actions):
        obs, state, reward, done, infos = self.env.step(key, state, actions)
        obs["__all__"]    = self.global_state(obs)
        reward["__all__"] = self.global_reward(reward)
        return obs, state, reward, done, infos

    @partial(jax.jit, static_argnums=0)
    def global_state(self, obs):
        return jnp.concatenate(list(obs.values()), axis=-1)
    
    @partial(jax.jit, static_argnums=0)
    def global_reward(self, reward):
        return jnp.stack(list(reward.values())).sum(axis=0) 
