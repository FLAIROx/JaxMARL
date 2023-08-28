""" NOT USED CURRENTLY"""

import jax 
import jax.numpy as jnp
from functools import partial
from typing import Tuple


# batch is used as the buffer
class BatchManagerIPPO:
    def __init__(
        self,
        #discount: float,
        #gae_lambda: float,
        n_steps: int,
        num_envs: int,
        num_agents: int,
        action_size,
        state_space,
    ):
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.action_size = action_size
        self.buffer_size = num_envs * n_steps
        self.num_actors = num_envs * num_agents
        self.n_steps = n_steps
        #self.discount = discount
        #self.gae_lambda = gae_lambda

        try:
            temp = state_space.shape[0]
            self.obs_shape = state_space.shape
        except Exception:
            self.obs_shape = [state_space]
        self.reset()

    @partial(jax.jit, static_argnums=0)
    def reset(self):
        return {
            "obs": jnp.empty(
                (self.n_steps, self.num_actors, *self.obs_shape),
                dtype=jnp.float32,
            ),
            "actions": jnp.empty(
                (self.n_steps, self.num_actors, *self.action_size),
            ),
            "rewards": jnp.empty(
                (self.n_steps, self.num_actors), dtype=jnp.float32
            ),
            "dones": jnp.empty((self.n_steps, self.num_actors), dtype=jnp.uint8),
            "terms": jnp.empty((self.n_steps, self.num_actors), dtype=jnp.uint8),
            "log_pis_old": jnp.empty(
                (self.n_steps, self.num_actors), dtype=jnp.float32
            ),
            "values_old": jnp.empty(
                (self.n_steps, self.num_actors), dtype=jnp.float32
            ),
            "_p": 0,
        }

    @partial(jax.jit, static_argnums=0)
    def append(self, buffer, state, action, reward, done, term, log_pi, value):
        return {
                "obs":  buffer["obs"].at[buffer["_p"]].set(state),
                "actions": buffer["actions"].at[buffer["_p"]].set(action),
                "rewards": buffer["rewards"].at[buffer["_p"]].set(reward.squeeze()),
                "dones": buffer["dones"].at[buffer["_p"]].set(done.squeeze()),
                "terms": buffer["terms"].at[buffer["_p"]].set(term.squeeze()),
                "log_pis_old": buffer["log_pis_old"].at[buffer["_p"]].set(log_pi),
                "values_old": buffer["values_old"].at[buffer["_p"]].set(value),
                "_p": (buffer["_p"] + 1) % self.n_steps,
            }

    @partial(jax.jit, static_argnums=0)
    def get(self, buffer):
        gae, target = self.calculate_gae(
            value=buffer["values_old"],
            reward=buffer["rewards"],
            done=buffer["dones"],
        )
        batch = (
            buffer["obs"][:-1],
            buffer["actions"][:-1],
            buffer["terms"][:-1],
            buffer["log_pis_old"][:-1],
            buffer["values_old"][:-1],
            target,
            gae,
        )
        return batch

    @partial(jax.jit, static_argnums=0)
    def calculate_gae(
        self, value: jnp.ndarray, reward: jnp.ndarray, done: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # NOTE this will take ages to compile
        advantages = []
        gae = 0.0
        for t in reversed(range(len(reward) - 1)):
            value_diff = self.discount * value[t + 1] * (1 - done[t]) - value[t]
            delta = reward[t] + value_diff
            gae = delta + self.discount * self.gae_lambda * (1 - done[t]) * gae
            advantages.append(gae)
        advantages = advantages[::-1]
        advantages = jnp.array(advantages)
        return advantages, advantages + value[:-1]
    
        