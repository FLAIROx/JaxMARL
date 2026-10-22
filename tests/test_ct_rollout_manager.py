"""
Test that CTRolloutManager zeroes the observations of agents that are done before the
episode ends, but not the first observation of the next episode after an auto-reset.
"""

import jax
import jax.numpy as jnp

from jaxmarl import make
from jaxmarl.environments.multi_agent_env import MultiAgentEnv, State
from jaxmarl.environments.spaces import Box, Discrete
from jaxmarl.wrappers.baselines import CTRolloutManager


class AgentDiesEarlyEnv(MultiAgentEnv):
    """Two agents; agent_0 is done after step 1, the episode ends after step 3.
    Every observation is the step count plus one, so none of them is zero."""

    def __init__(self):
        super().__init__(num_agents=2)
        self.observation_spaces = {a: Box(-jnp.inf, jnp.inf, (3,)) for a in self.agents}
        self.action_spaces = {a: Discrete(2) for a in self.agents}

    def reset(self, key):
        state = State(done=jnp.array(False), step=jnp.array(0))
        return self.get_obs(state), state

    def get_obs(self, state):
        return {a: jnp.full((3,), state.step + 1.0) for a in self.agents}

    def step_env(self, key, state, actions):
        step = state.step + 1
        state = State(done=step >= 3, step=step)
        dones = {"agent_0": step >= 1, "agent_1": step >= 3, "__all__": step >= 3}
        rewards = {a: jnp.array(0.0) for a in self.agents}
        return self.get_obs(state), state, rewards, dones, {}


def _step(wrapper, key, state):
    actions = {a: jnp.array(0) for a in wrapper.agents}
    return wrapper.wrapped_step(key, state, actions)


def test_agent_done_before_episode_end_gets_zero_obs():
    wrapper = CTRolloutManager(AgentDiesEarlyEnv(), batch_size=1)
    key = jax.random.PRNGKey(0)
    _, state = wrapper.wrapped_reset(key)
    obs, state, _, dones, _ = _step(wrapper, key, state)
    assert dones["agent_0"] and not dones["__all__"]
    assert jnp.all(obs["agent_0"] == 0)
    assert jnp.all(obs["agent_1"][:3] == 2.0)


def test_first_obs_of_next_episode_is_kept_after_auto_reset():
    wrapper = CTRolloutManager(AgentDiesEarlyEnv(), batch_size=1)
    key = jax.random.PRNGKey(0)
    reset_obs, state = wrapper.wrapped_reset(key)
    for _ in range(3):
        obs, state, _, dones, _ = _step(wrapper, key, state)
    assert dones["__all__"]
    for agent in wrapper.agents:
        assert jnp.array_equal(obs[agent], reset_obs[agent])


def test_mpe_first_obs_of_next_episode_is_kept_after_auto_reset():
    env = make("MPE_simple_spread_v3")
    wrapper = CTRolloutManager(env, batch_size=1)
    key, key_reset = jax.random.split(jax.random.PRNGKey(0))
    _, state = wrapper.wrapped_reset(key_reset)
    for _ in range(env.max_steps + 1):
        key, key_step, key_act = jax.random.split(key, 3)
        key_act = jax.random.split(key_act, env.num_agents)
        actions = {
            a: env.action_space(a).sample(key_act[i]) for i, a in enumerate(env.agents)
        }
        obs, state, _, dones, _ = wrapper.wrapped_step(key_step, state, actions)
        if dones["__all__"]:
            break
    assert dones["__all__"]
    first_obs = env.get_obs(state)  # the state has already been reset
    for agent in env.agents:
        expected = wrapper._preprocess_obs(
            first_obs[agent], wrapper.agents_one_hot[agent]
        )
        assert jnp.any(expected != 0)
        assert jnp.array_equal(obs[agent], expected)
