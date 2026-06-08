"""
Abstract base class for multi agent gym environments with JAX
Based on the Gymnax and PettingZoo APIs
"""

from functools import partial
from typing import Any, Dict, Optional, Tuple

import jax
from flax import struct
from jaxtyping import Array, Bool, Float, Int, Num, PRNGKeyArray

from jaxmarl.environments.spaces import Space

Observations = Dict[str, Num[Array, "..."]]
Actions = Dict[str, Num[Array, "..."]]
Rewards = Dict[str, Float[Array, ""]]
Dones = Dict[str, Bool[Array, ""]]
Infos = Dict[str, Any]
AvailActions = Dict[str, Bool[Array, "..."]]


@struct.dataclass
class State:
    """Base environment state.

    Environment-specific states should usually subclass or replace this with
    additional JAX-compatible fields.

    Attributes:
        done: Whether the episode has terminated.
        step: Current environment step count.
    """

    done: Bool[Array, ""]
    step: Int[Array, ""]


class MultiAgentEnv(object):
    """Jittable abstract base class for all JaxMARL Environments.

    Subclasses should implement ``reset``, ``step_env``, ``get_obs``,
    ``get_avail_actions``, and ``agent_classes``. The public ``step`` method
    handles automatic reset when ``dones["__all__"]`` is true.
    """

    def __init__(
        self,
        num_agents: int,
    ) -> None:
        """Initialise the multi-agent environment.

        Args:
            num_agents: Maximum number of agents in the environment. This is
                commonly used to define array dimensions and agent metadata.
        """
        self.num_agents = num_agents
        self.agents = ["agent_{}".format(i) for i in range(num_agents)]
        self.observation_spaces = dict()
        self.action_spaces = dict()

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: PRNGKeyArray) -> Tuple[Observations, State]:
        """Reset the environment.

        Args:
            key: Random key used to initialise the environment state.

        Returns:
            A tuple containing the initial observations and initial environment
            state.
        """
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: PRNGKeyArray,
        state: State,
        actions: Actions,
        reset_state: Optional[State] = None,
    ) -> Tuple[Observations, State, Rewards, Dones, Infos]:
        """Performs step transitions in the environment. Resets the environment if done.
        To control the reset state, pass `reset_state`. Otherwise, the environment will reset using `self.reset`.

        Args:
            key: Random key.
            state: Current environment state.
            actions: Agent actions, keyed by agent name.
            reset_state: Optional environment state to reset to on episode completion.

        Returns:
            Tuple containing next observations, next state, rewards, dones, and info.
        """

        key, key_reset = jax.random.split(key)
        obs_st, states_st, rewards, dones, infos = self.step_env(key, state, actions)

        if reset_state is None:
            obs_re, states_re = self.reset(key_reset)
        else:
            states_re = reset_state
            obs_re = self.get_obs(states_re)

        # Auto-reset environment based on termination
        states = jax.tree.map(
            lambda x, y: jax.lax.select(dones["__all__"], x, y), states_re, states_st
        )
        obs = jax.tree.map(
            lambda x, y: jax.lax.select(dones["__all__"], x, y), obs_re, obs_st
        )
        return obs, states, rewards, dones, infos

    def step_env(
        self, key: PRNGKeyArray, state: State, actions: Actions
    ) -> Tuple[Observations, State, Rewards, Dones, Infos]:
        """Perform one environment-specific transition.

        Subclasses should implement this method with the actual transition
        logic. Unlike ``step``, this method should not perform automatic reset.

        Args:
            key: Random key used for the transition.
            state: Current environment state.
            actions: Agent actions, keyed by agent name.

        Returns:
            A tuple containing next observations, next state, rewards, dones,
            and auxiliary info.
        """

        raise NotImplementedError

    def get_obs(self, state: State) -> Observations:
        """Applies observation function to state.

        Args:
            state: Environment state.

        Returns:
            Observations keyed by agent name.
        """
        raise NotImplementedError

    def observation_space(self, agent: str) -> Space:
        """Observation space for a given agent.

        Args:
            agent: Agent name.

        Returns:
            The observation space for the requested agent.
        """
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> Space:
        """Action space for a given agent.

        Args:
            agent: Agent name.

        Returns:
            The action space for the requested agent.
        """
        return self.action_spaces[agent]

    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: State) -> AvailActions:
        """Returns the available actions for each agent.

        Args:
            state: Environment state.

        Returns:
            Available actions keyed by agent name. Values are boolean
            masks, where true entries indicate actions that are currently
            available.
        """
        raise NotImplementedError

    @property
    def name(self) -> str:
        """Environment name."""
        return type(self).__name__

    @property
    def agent_classes(self) -> dict:
        """Return agent class metadata.

        Returns:
            A mapping from class names to constituent agent names. Environments can
            use this to group agents with shared policies or roles.

        Example:
            ``{"adversary": ["agent_0", "agent_2"], "good": ["agent_1"]}``
        """
        raise NotImplementedError
