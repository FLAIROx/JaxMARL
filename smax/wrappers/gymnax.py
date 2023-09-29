import jax
import jax.numpy as jnp
from functools import partial
import gymnax

class GymnaxToSMAX(object):
    
    
    def __init__(self, env_name: str, num_agents: int = 1, env_kwargs: dict = {}):
        self.env_name = env_name
        self.num_agents = num_agents
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]
        self._env, self.env_params = gymnax.make(env_name, **env_kwargs)
        self.step_fn = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))
        self.reset_fn = jax.vmap(self._env.reset, in_axes=(0, None))
        
    @property
    def default_params(self):
        return self.env_params
        
    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, actions, params=None):
        keys = jax.random.split(key, num=self.num_agents)
        actions = jnp.stack([actions[agent] for agent in self.agents])
        obs, state, reward, done, info = self.step_fn(keys, state, actions.squeeze(axis=-1), params)
        obs = {agent: obs[i] for i, agent in enumerate(self.agents)}
        reward = {agent: reward.mean() for agent in self.agents}
        smax_dones = {agent: done[i] for i, agent in enumerate(self.agents)}
        smax_dones["__all__"] = jnp.all(done)
        return obs, state, reward, smax_dones, info
    
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        keys = jax.random.split(key, num=self.num_agents)
        obs, state = self.reset_fn(keys, params)
        obs = {agent: obs[i] for i, agent in enumerate(self.agents)}
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state):
        return {agent: jnp.ones((self.action_space(agent).n,)) for agent in self.agents}

    def observation_space(self, agent: str):
        return self._env.observation_space(self.env_params)
    
    def action_space(self, agent: str):
        return self._env.action_space(self.env_params)
    
    