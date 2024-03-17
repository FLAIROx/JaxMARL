"""
Investment environment based on https://doi.org/10.1038/s41562-022-01383-x
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import chex
from flax import struct
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.environments.spaces import Discrete, MultiDiscrete
from functools import partial


@struct.dataclass
class State:
    """
    State of investment game
    """

    agents_money: chex.Array
    contributions: chex.Array
    step: int


class InvestmentEnv(MultiAgentEnv):
    """Represents investment environment"""

    def __init__(
            self,
            num_agents=4,
            num_rounds=1,
            seed=0,
            v=1,
            w=1
            ):
        super().__init__(num_agents=4)
        key = jax.random.PRNGKey(seed)
        _, key_head, key_endowment = jax.random.split(key, 3)

        # Agents
        self.num_agents = num_agents
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]

        # Rounds
        self.num_rounds = num_rounds

        # Endowments
        # the amount of money a player receives each round
        head = jax.random.choice(key_head, jnp.arange(self.num_agents))
        self.endowments = jnp.repeat(jax.random.randint(key_endowment, (1,), 2, 9), repeats=self.num_agents)
        self.endowments = self.endowments.at[head].set(10)

        # Action spaces
        self.action_spaces = {a: Discrete(10) for a, e in zip(self.agents, self.endowments)}

        # Observation spaces
        self.observation_spaces = {
            a: MultiDiscrete([10] * (2 * self.num_agents))
            for a in self.agents
            }

        # Manifold
        self.v = v
        self.w = w
        self.r = 1.6

    @partial(jax.jit, static_argnums=[0])
    def reset(self, key):
        """Performs resetting of environment

        Returns: obs, state
        """
        state = State(
            agents_money=self.endowments,
            contributions=jnp.zeros(self.num_agents, dtype=jnp.int32),
            step=1
            )
        return self.get_obs(state), state

    @partial(jax.jit, static_argnums=[0])
    def step_env(self, key, state, actions):
        """Performs step transitions in the environment

        Returns: obs, state, rewards, done, info
        """
        # Get the actions as array
        actions = jnp.array([actions[i] for i in self.agents]).reshape((self.num_agents,))

        # Common pot
        common_pot = jnp.sum(actions)

        # Find ratios
        contribution_ratios = actions / state.agents_money
        tot_ratio = common_pot / jnp.sum(state.agents_money)

        # Find other rewards
        other_rewards = jnp.repeat(actions.reshape([1, self.num_agents]), self.num_agents, axis=0).reshape(self.num_agents, self.num_agents)
        di = jnp.diag_indices(self.num_agents)
        other_rewards = other_rewards.at[di].set(0)

        # Find other ratios
        ro = actions / self.endowments
        other_ratios = jnp.repeat(contribution_ratios.reshape([1, self.num_agents]), self.num_agents, axis=0).reshape(self.num_agents, self.num_agents)
        other_ratios = other_ratios.at[di].set(0)

        # Find y
        y_abs = self.r * (self.w * actions + (1 - self.w) * jnp.mean(other_rewards, axis=1))
        y_rel = self.r * (common_pot/tot_ratio) * (self.w * ro + (1 - self.w) * jnp.mean(other_ratios, axis=1)) # TODO should these be a mean??
        y = self.v * y_rel + (1 - self.v) * y_abs

        # Update the environment state
        step = state.step + 1
        state = State(
            agents_money=self.endowments,
            contributions=actions,
            step=step
            )

        # Prepare outputs
        obs = self.get_obs(state)
        rewards = y - actions + self.endowments
        rewards = {a: rewards[i] for a, i in zip(self.agents, range(self.num_agents))}
        done = self.is_terminal(state)
        dones = {a: done for a in self.agents + ["__all__"]}
        info = {}

        return (obs, state, rewards, dones, info)

    @partial(jax.jit, static_argnums=[0])
    def get_obs(self, state):
        """Applies observation function to state.

        Returns: obs
        """
        obs = jnp.concatenate((state.agents_money, state.contributions)).astype(jnp.int32)

        return {a: obs for a in self.agents}

    def is_terminal(self, state):
        """Check whether state is terminal

        Returns: is_terminal
        """
        is_terminal = state.step > self.num_rounds

        return is_terminal

    def render(self, state):
        """Renders environment on command line

        Returns: N/A
        """
        print(f"\nCurrent step: {state.step}")
        print(f"Agents' money: {state.agents_money}")
        print(f"Contributions: {state.contributions}")


    @property
    def name(self) -> str:
        """Environment name

        Returns: env_name
        """
        return "EconomicsEnv"

    @property
    def agent_classes(self) -> dict:
        """Returns a dictionary with agent classes, used in environments with hetrogenous agents

        Format:
            agent_base_name: [agent_base_name_1, agent_base_name_2, ...]
        """
        raise NotImplementedError


def example():
    """Simple manipulations of environment for debugging

    Returns: N/A
    """
    key = jax.random.PRNGKey(0)

    env = InvestmentEnv()

    _, state = env.reset(key)

    while not env.is_terminal(state):
        key, key_step, key_act = jax.random.split(key, 3)

        # Render environment state
        env.render(state)

        # Sample random actions
        key_act = jax.random.split(key_act, env.num_agents)
        actions = {
            a: env.action_spaces[a].sample(key_act[i])
            for i, a in enumerate(env.agents)
            }

        # Print action
        print("Actions:", actions)

        # Perform step in environment
        obs, state, rewards, done, _ = env.step_env(key_step, state, actions)

        # Print metadata
        print(f"Observation: {obs}")
        print(f"Rewards: {rewards}")
        print(f"Done: {done}")


if __name__ == "__main__":
    example()
