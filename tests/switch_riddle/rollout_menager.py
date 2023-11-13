from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from functools import partial
import jax


class RolloutManager:
    def __init__(self, env: MultiAgentEnv, batch_size: int):
        self.env = env
        self.batch_size = batch_size
        self.batch_samplers = {
            agent: jax.jit(jax.vmap(env.action_space(agent).sample, in_axes=0))
            for agent in self.env.agents
        }

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
