import jax
import jax.numpy as jnp
from functools import partial
from jaxmarl.environments.multi_agent_env import MultiAgentEnv

class RolloutManager:

    def __init__(self, env: MultiAgentEnv):

        self.env = env
    
    @partial(jax.jit, static_argnums=[0])
    def batch_reset(self, keys):
        return jax.vmap(self.env.reset_env, in_axes=(0,))(keys)
    
    @partial(jax.jit, static_argnums=0)
    def batch_step(self, keys, states, actions):
        return jax.vmap(self.env.step, in_axes=(0, 0, 0))(keys, states, actions)