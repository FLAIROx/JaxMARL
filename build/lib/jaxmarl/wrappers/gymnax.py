import jax
from functools import partial
import gymnax

class GymnaxToJaxMARL(object):
    """ Use a Gymnax Environment within JaxMARL. Useful for debugging """
    
    num_agents = 1
    agent = "agent"
    agents = [agent]
    
    def __init__(self, env_name: str, env_kwargs: dict = {}):
        self.env_name = env_name 
        self._env, self.env_params = gymnax.make(env_name, **env_kwargs)
        
    @property
    def default_params(self):
        return self.env_params
        
    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, actions, params=None):
        print('act', actions[self.agent])
        obs, state, reward, done, info = self._env.step(key, state, actions[self.agent].squeeze(), params)
        obs = {self.agent: obs}
        reward = {self.agent: reward}
        done = {self.agent: done, "__all__": done}
        return obs, state, reward, done, info
    
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        obs = {self.agent: obs}
        return obs, state
        
    def observation_space(self, agent: str):
        return self._env.observation_space(self.env_params)
    
    def action_space(self, agent: str):
        return self._env.action_space(self.env_params)
    
    