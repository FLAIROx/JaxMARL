import torch
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
import jax
import jax.numpy as jnp
from jax import lax
from gymnax.environments import environment, spaces
from typing import Tuple, Optional
import chex
from flax import struct
import haiku as hk


@dataclass
class EnvState:
    agents_money: ...
    contributions: ...
    time: ...

@dataclass
class EnvParams:
    nothing: None


class InvestmentEnv(environment.Environment):
    def __init__(self, seed, v, w):
        super().__init__()
        rng = jax.random.PRNGKey(seed)
        rng_head, rng_endowment, key = jax.random.split(rng, 3)
        # choosing head player
        head = jax.random.choice(rng_head, jnp.array([0,1,2,3]))
        # choosing value of endowment for all players rather than head
        self.endowment = jnp.repeat(jnp.random.randint(rng_endowment, 2, 9), repeats=4)
        # assigning endowment of head player to 10
        self.endowment[head] = 10
        self.v = v
        self.w = w
        self.r = 1.6
        # self.state = EnvState(agents_money=self.endowment, contributions=jnp.zeros(4), time=0)

    def step_env(
        self, key, state, actions, params = None
    ):
        """Performs step transitions in the environment.

        Returns: env_state, obsv, reward, done, info
        """
        common_pot = jnp.sum(actions)
        contribution_ratios = actions / state.agents_money
        tot_ratio = common_pot / jnp.sum(state.agents_money)
        other_rewards = jnp.repeat(actions.reshape([1, 4]), 4, axis=0).reshape(4,4)
        di = jnp.diag_indices(4)
        other_rewards = other_rewards.at[di].set(0)
        y_abs = self.r * (self.w * actions + (1 - self.w) * jnp.mean(other_rewards, axis=1))
        p = actions / contribution_ratios
        other_ratios = jnp.repeat(p.reshape([1, 4]), 4, axis=0).reshape(4,4)
        other_ratios = other_ratios.at[di].set(0)
        y_rel = self.r * (common_pot/tot_ratio) * (self.w * p + (1 - self.w) * jnp.mean(other_ratios, axis=1)) # TODO should these be a mean??
        y = self.v * y_rel + (1 - self.v) * y_abs
        self.state = EnvState(agents_money=self.endowment, contributions=actions, time=self.state.time + 1)
        done = self.is_terminal(self.state)
        obs = self.get_obs(self.state)
        selfish_rewards = (y-actions+self.endowment)
        return (
            obs,
            selfish_rewards,
            done,
            {"state": self.state},
        )

    def reset_env(
        self, key, params = None,
    ):
        """Performs resetting of environment.

        Returns: state, obs
        """
        self.state = EnvState(agents_money=self.endowment, contributions=jnp.zeros(4), time=0)
        return self.get_obs(self.state)

    def get_obs(self, state):
        """Applies observation function to state."""

        return jnp.concatenate((state.agents_money, state.contributions))

    def is_terminal(self, state):
        """Check whether state is terminal."""
 
        return state.time >= 10

    @property
    def name(self) -> str:
        """Environment name."""
        return "EconomicsEnv"