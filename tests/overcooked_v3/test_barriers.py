"""Tests for the barrier system in Overcooked V3."""

import jax
import jax.numpy as jnp
import pytest
from jaxmarl import make
from jaxmarl.environments.overcooked_v3.common import Actions


class TestBarriers:
    """Test that barriers block movement when active and allow it when inactive."""

    def _make_env_and_reset(self, layout, **kwargs):
        """Helper: create barrier env with given layout and reset it."""
        env = make("overcooked_v3", layout=layout, **kwargs)
        key = jax.random.PRNGKey(0)
        key, reset_key = jax.random.split(key)
        obs, state = env.reset(reset_key)
        return env, key, obs, state

    def test_inactive_barrier_allows_movement(self):
        """Agent can move through an inactive barrier."""
        env, key, obs, state = self._make_env_and_reset("barrier_demo")

        # Explicitly set barriers to inactive
        state = state.replace(
            barrier_active=jnp.zeros_like(state.barrier_active, dtype=jnp.bool_)
        )

        # Move agent_0 right to the cell adjacent to the barrier, then through it.
        actions_right = {"agent_0": int(Actions.right), "agent_1": int(Actions.stay)}
        key, step_key = jax.random.split(key)
        _, state, _, _, _ = env.step_env(step_key, state, actions_right)

        key, step_key = jax.random.split(key)
        _, new_state, _, _, _ = env.step_env(step_key, state, actions_right)

        assert new_state.agents.pos.x[0] == 3
        assert new_state.agents.pos.y[0] == 1

    def test_active_barrier_blocks_movement(self):
        """Agent cannot move through an active barrier."""
        env, key, obs, state = self._make_env_and_reset("barrier_demo")

        # Activate all barriers
        state = state.replace(
            barrier_active=jnp.ones_like(state.barrier_active, dtype=jnp.bool_)
        )

        # Move agent_0 right twice to get adjacent to barrier, then try to cross
        actions_right = {"agent_0": 0, "agent_1": 4}
        for _ in range(3):
            key, step_key = jax.random.split(key)
            obs, state, rewards, dones, info = env.step_env(
                step_key, state, actions_right
            )

        # Record position and try one more move into the barrier
        pos_before_x = state.agents.pos.x[0]
        pos_before_y = state.agents.pos.y[0]

        key, step_key = jax.random.split(key)
        obs, state, rewards, dones, info = env.step_env(step_key, state, actions_right)

        assert state.agents.pos.x[0] == pos_before_x, (
            "Agent x-position should not change when blocked by active barrier"
        )
        assert state.agents.pos.y[0] == pos_before_y, (
            "Agent y-position should not change when blocked by active barrier"
        )

    def test_deactivated_barrier_allows_movement(self):
        """After deactivating a barrier, agent can move through it."""
        env, key, obs, state = self._make_env_and_reset("barrier_demo")

        # Activate barriers, then deactivate them
        state = state.replace(
            barrier_active=jnp.ones_like(state.barrier_active, dtype=jnp.bool_)
        )
        state = state.replace(
            barrier_active=jnp.zeros_like(state.barrier_active, dtype=jnp.bool_)
        )

        # Move agent_0 right toward barrier position
        actions_right = {"agent_0": 0, "agent_1": 4}
        positions = []
        for _ in range(5):
            key, step_key = jax.random.split(key)
            obs, state, rewards, dones, info = env.step_env(
                step_key, state, actions_right
            )
            positions.append(int(state.agents.pos.x[0]))

        # Agent should have moved at least once (not stuck)
        assert len(set(positions)) > 1, (
            "Agent should be able to move through deactivated barrier"
        )


class TestTimedBarriers:
    """Test timed barrier functionality with button interaction."""

    BARRIER_DURATION = 5

    def _make_env_and_reset(self):
        """Helper: create timed barrier env and reset it."""
        env = make(
            "overcooked_v3",
            layout="timed_barrier_demo",
            barrier_duration=self.BARRIER_DURATION,
        )
        key = jax.random.PRNGKey(0)
        key, reset_key = jax.random.split(key)
        obs, state = env.reset(reset_key)
        return env, key, obs, state

    def _navigate_to_button_and_press(self, env, key, state):
        """Helper: move agent to button and press it. Returns updated key and state."""
        # Move toward barrier first
        actions_right = {"agent_0": 0, "agent_1": 4}
        for _ in range(2):
            key, step_key = jax.random.split(key)
            _, state, _, _, _ = env.step_env(step_key, state, actions_right)

        # Move down to button
        actions_down = {"agent_0": 1, "agent_1": 4}
        key, step_key = jax.random.split(key)
        _, state, _, _, _ = env.step_env(step_key, state, actions_down)

        # Press button
        actions_interact = {"agent_0": 5, "agent_1": 4}
        key, step_key = jax.random.split(key)
        _, state, _, _, _ = env.step_env(step_key, state, actions_interact)

        return key, state

    def test_active_barrier_blocks(self):
        """Active timed barrier blocks movement."""
        env, key, obs, state = self._make_env_and_reset()

        # Explicitly set barrier to active
        state = state.replace(
            barrier_active=jnp.ones_like(state.barrier_active, dtype=jnp.bool_)
        )

        # Try moving toward barrier
        actions_right = {"agent_0": 0, "agent_1": 4}
        for _ in range(2):
            key, step_key = jax.random.split(key)
            _, state, _, _, _ = env.step_env(step_key, state, actions_right)

        pos_before_x = state.agents.pos.x[0]
        key, step_key = jax.random.split(key)
        _, state, _, _, _ = env.step_env(step_key, state, actions_right)

        assert state.agents.pos.x[0] == pos_before_x, (
            "Agent should be blocked by active timed barrier"
        )

    def test_button_deactivates_barrier_and_sets_timer(self):
        """Pressing button deactivates barrier and starts countdown timer."""
        env, key, obs, state = self._make_env_and_reset()

        key, state = self._navigate_to_button_and_press(env, key, state)

        assert not state.barrier_active[0], (
            "Barrier should be inactive after button press"
        )

        expected_timer = int(state.barrier_duration[0]) - 1
        assert state.barrier_timer[0] == expected_timer, (
            f"Timer should be barrier_duration - 1 = {expected_timer} "
            f"(decremented on same step), got {state.barrier_timer[0]}"
        )

    def test_deactivated_barrier_allows_movement(self):
        """Agent can move through a deactivated timed barrier."""
        env, key, obs, state = self._make_env_and_reset()

        key, state = self._navigate_to_button_and_press(env, key, state)

        pos_before_x = state.agents.pos.x[0]
        actions_right = {"agent_0": 0, "agent_1": 4}
        key, step_key = jax.random.split(key)
        _, state, _, _, _ = env.step_env(step_key, state, actions_right)

        assert state.agents.pos.x[0] != pos_before_x, (
            "Agent should move through deactivated barrier"
        )

    def test_barrier_reactivates_after_timer_expires(self):
        """Barrier reactivates once the countdown timer reaches zero."""
        env, key, obs, state = self._make_env_and_reset()

        key, state = self._navigate_to_button_and_press(env, key, state)

        assert not state.barrier_active[0], "Barrier should be inactive after press"

        # Step until timer expires
        steps_to_simulate = int(state.barrier_timer[0])
        actions_stay = {"agent_0": 4, "agent_1": 4}
        for _ in range(steps_to_simulate):
            key, step_key = jax.random.split(key)
            _, state, _, _, _ = env.step_env(step_key, state, actions_stay)

        assert state.barrier_active[0], "Barrier should reactivate after timer expires"
        assert state.barrier_timer[0] == 0, "Timer should be 0 after expiration"

    def test_reactivated_barrier_blocks(self):
        """After reactivation, barrier blocks movement again."""
        env, key, obs, state = self._make_env_and_reset()

        key, state = self._navigate_to_button_and_press(env, key, state)

        # Wait for timer to expire
        steps_to_simulate = int(state.barrier_timer[0])
        actions_stay = {"agent_0": 4, "agent_1": 4}
        for _ in range(steps_to_simulate):
            key, step_key = jax.random.split(key)
            _, state, _, _, _ = env.step_env(step_key, state, actions_stay)

        assert state.barrier_active[0], "Barrier should have reactivated"

        # Agent is adjacent to the reactivated barrier; trying to enter it is blocked.
        pos_before_x = state.agents.pos.x[0]
        pos_before_y = state.agents.pos.y[0]
        actions_right = {"agent_0": int(Actions.right), "agent_1": int(Actions.stay)}
        key, step_key = jax.random.split(key)
        _, state, _, _, _ = env.step_env(step_key, state, actions_right)

        assert state.agents.pos.x[0] == pos_before_x, (
            "Agent should be blocked by reactivated barrier"
        )
        assert state.agents.pos.y[0] == pos_before_y, (
            "Agent should be blocked by reactivated barrier"
        )
