"""
Investment environment based on https://doi.org/10.1038/s41562-022-01383-x
"""
from __future__ import annotations

import jax
from flax import struct
from jaxmarl.environments.multi_agent_env import MultiAgentEnv


@struct.dataclass
class State:
    """
    State of investment game
    """

    step: int


class InvestmentEnv(MultiAgentEnv):
    """Represents investment environment"""

    def __init__(
            self,
            v=1,
            w=1
            ):
        super().__init__(num_agents=4)
        self.v = v
        self.w = w

    def reset(self, key):
        """Performs resetting of environment

        Returns: state, obs
        """
        state = State(
            step=0
            )
        return self.get_obs(state), state

    def step_env(self, key, state, actions):
        """Performs step transitions in the environment

        Returns: obs, state, rewards, dones, info
        """
        step = state.step + 1

        obs = None
        state = State(
            step=step
            )
        rewards = None
        dones = None
        info = None

        return (obs, state, rewards, dones, info)

    def get_obs(self, state):
        """Applies observation function to state.

        Returns: obs
        """
        obs = None

        return obs

    def is_terminal(self, state):
        """Check whether state is terminal

        Returns: is_terminal
        """
        is_terminal = None

        return is_terminal

    def render(self, state):
        """Renders environment on command line

        Returns: N/A
        """
        print(f"\nCurrent step: {state.step}")

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
    env.render(state)


if __name__ == "__main__":
    example()
