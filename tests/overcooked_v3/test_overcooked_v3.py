"""Tests for Overcooked V3 environment."""

import jax
import jax.numpy as jnp
import pytest
from jaxmarl import make
from jaxmarl.environments.overcooked_v3 import OvercookedV3, overcooked_v3_layouts
from jaxmarl.environments.overcooked_v3.common import (
    Actions,
    Direction,
    DynamicObject,
    Position,
    StaticObject,
)
from jaxmarl.environments.overcooked_v3.layouts import (
    Layout,
    coordinated_temporal_conveyor,
)
from jaxmarl.environments.multi_agent_env import MultiAgentEnv


class TestOvercookedV3API:
    """Test that OvercookedV3 implements the MultiAgentEnv interface."""

    def test_inherits_base_env(self):
        env = OvercookedV3()
        assert isinstance(env, MultiAgentEnv)

    def test_has_required_attributes(self):
        env = OvercookedV3()
        assert hasattr(env, 'agents')
        assert hasattr(env, 'num_agents')
        assert hasattr(env, 'action_spaces')
        assert hasattr(env, 'observation_spaces')

    def test_reset_returns_correct_format(self):
        env = OvercookedV3()
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)

        # Check obs is dict with all agent keys
        assert isinstance(obs, dict)
        for agent in env.agents:
            assert agent in obs

        # Check state has required fields
        assert hasattr(state, 'time')
        assert hasattr(state, 'terminal')
        assert hasattr(state, 'grid')
        assert hasattr(state, 'agents')

    def test_step_returns_correct_format(self):
        env = OvercookedV3()
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)

        # Random actions
        actions = {agent: 0 for agent in env.agents}
        key, subkey = jax.random.split(key)
        obs, state, rewards, dones, info = env.step(subkey, state, actions)

        # Check dones has __all__ key
        assert "__all__" in dones

        # Check all agents have entries
        for agent in env.agents:
            assert agent in obs
            assert agent in rewards
            assert agent in dones


class TestOvercookedV3Rollout:
    """Test that random rollouts work correctly."""

    def test_random_rollout_completes(self):
        """Run a full episode with random actions."""
        env = OvercookedV3(max_steps=100)
        key = jax.random.PRNGKey(42)

        key, subkey = jax.random.split(key)
        obs, state = env.reset(subkey)

        done = False
        step_count = 0
        total_reward = 0.0

        while not done and step_count < 200:
            # Random actions for all agents
            key, *subkeys = jax.random.split(key, len(env.agents) + 1)
            actions = {
                agent: int(jax.random.randint(subkeys[i], (), 0, 6))
                for i, agent in enumerate(env.agents)
            }

            key, subkey = jax.random.split(key)
            obs, state, rewards, dones, info = env.step(subkey, state, actions)

            total_reward += sum(rewards.values())
            done = dones["__all__"]
            step_count += 1

        assert step_count > 0
        print(f"Rollout completed in {step_count} steps, total reward: {total_reward}")

    def test_jit_compiled_rollout(self):
        """Verify rollout works with JIT compilation."""
        env = OvercookedV3(max_steps=50)

        @jax.jit
        def step_fn(key, state, actions):
            return env.step(key, state, actions)

        @jax.jit
        def reset_fn(key):
            return env.reset(key)

        key = jax.random.PRNGKey(0)
        key, subkey = jax.random.split(key)
        obs, state = reset_fn(subkey)

        for _ in range(10):
            actions = {agent: 0 for agent in env.agents}
            key, subkey = jax.random.split(key)
            obs, state, rewards, dones, info = step_fn(subkey, state, actions)

        assert True  # If we get here without error, JIT works

    def test_vmap_parallel_envs(self):
        """Verify environment works with vmap for parallel rollouts."""
        env = OvercookedV3()
        num_envs = 4

        keys = jax.random.split(jax.random.PRNGKey(0), num_envs)
        reset_fn = jax.vmap(env.reset)
        obs, states = reset_fn(keys)

        # Check batched dimensions
        assert obs[env.agents[0]].shape[0] == num_envs


class TestOvercookedV3Layouts:
    """Test layout loading and parsing."""

    def test_cramped_room_layout(self):
        env = OvercookedV3(layout="cramped_room")
        assert env.num_agents == 2
        assert env.height > 0
        assert env.width > 0

    def test_all_registered_layouts(self):
        """Test that all registered layouts can be loaded."""
        for layout_name in overcooked_v3_layouts.keys():
            env = OvercookedV3(layout=layout_name)
            key = jax.random.PRNGKey(0)
            obs, state = env.reset(key)
            assert obs is not None
            assert state is not None


class TestOvercookedV3Interactions:
    """Test interaction edge cases."""

    def test_interact_left_edge_does_not_wrap_to_right_edge(self):
        layout = Layout.from_string(
            """
A BP 0
W   XW
WWWWWW
""",
            possible_recipes=[[0, 0, 0]],
        )
        env = OvercookedV3(layout=layout, random_agent_positions=False)
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)

        right_col = env.width - 1
        assert StaticObject.is_ingredient_pile(state.grid[0, right_col, 0])

        state = state.replace(
            agents=state.agents.replace(
                pos=state.agents.pos.replace(
                    x=state.agents.pos.x.at[0].set(0),
                    y=state.agents.pos.y.at[0].set(0),
                ),
                dir=state.agents.dir.at[0].set(Direction.LEFT),
                inventory=state.agents.inventory.at[0].set(0),
            )
        )

        actions = {"agent_0": int(Actions.interact)}
        obs, new_state, rewards, dones, info = env.step(
            jax.random.PRNGKey(1), state, actions
        )

        assert new_state.agents.inventory[0] == 0


class TestOvercookedV3PotMechanics:
    """Test pot cooking and burning mechanics."""

    def test_pot_initial_state(self):
        """Verify pots start empty."""
        env = OvercookedV3()
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)

        # All pot timers should be 0
        assert jnp.all(state.pot_cooking_timer == 0)

    def _setup_full_pot(self, env, state, timer_value):
        """Helper: set pot 0 to 3 onions with a given timer."""
        from jaxmarl.environments.overcooked_v3.common import DynamicObject
        pot_y, pot_x = state.pot_positions[0]
        full_pot = DynamicObject.ingredient(0) * 3  # 3 onions
        new_grid = state.grid.at[pot_y, pot_x, 1].set(full_pot)
        new_timers = state.pot_cooking_timer.at[0].set(timer_value)
        return state.replace(grid=new_grid, pot_cooking_timer=new_timers)

    def _step_noop(self, env, state, key):
        """Helper: take a no-op step for all agents."""
        actions = {agent: 4 for agent in env.agents}
        key, subkey = jax.random.split(key)
        obs, new_state, rewards, dones, info = env.step(subkey, state, actions)
        return new_state, key

    def test_pot_cooking_timer_decrements(self):
        """Verify pot cooking timer decrements when pot is full."""
        env = OvercookedV3(pot_cook_time=10, pot_burn_time=5)
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)

        state = self._setup_full_pot(env, state, timer_value=10)

        new_state, key = self._step_noop(env, state, key)

        # Timer should have decremented
        assert new_state.pot_cooking_timer[0] == 9

    def test_pot_becomes_cooked_at_burn_time(self):
        """Verify pot gets COOKED flag when timer reaches burn_time."""
        from jaxmarl.environments.overcooked_v3.common import DynamicObject
        env = OvercookedV3(pot_cook_time=10, pot_burn_time=5)
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)

        # Set timer to burn_time + 1, so one step brings it to burn_time
        state = self._setup_full_pot(env, state, timer_value=6)

        new_state, key = self._step_noop(env, state, key)

        # Timer should now be at burn_time (5)
        assert new_state.pot_cooking_timer[0] == 5

        # Pot should have COOKED flag set
        pot_y, pot_x = new_state.pot_positions[0]
        pot_ingredients = new_state.grid[pot_y, pot_x, 1]
        assert (pot_ingredients & DynamicObject.COOKED) != 0

    def test_pot_timer_continues_after_cooked(self):
        """Verify timer keeps decrementing in the burn window after COOKED is set."""
        from jaxmarl.environments.overcooked_v3.common import DynamicObject
        env = OvercookedV3(pot_cook_time=10, pot_burn_time=5)
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)

        # Set up a pot that's already cooked (COOKED flag + timer in burn window)
        pot_y, pot_x = state.pot_positions[0]
        cooked_pot = DynamicObject.ingredient(0) * 3 | DynamicObject.COOKED
        new_grid = state.grid.at[pot_y, pot_x, 1].set(cooked_pot)
        new_timers = state.pot_cooking_timer.at[0].set(4)
        state = state.replace(grid=new_grid, pot_cooking_timer=new_timers)

        new_state, key = self._step_noop(env, state, key)

        # Timer should have decremented (burn window still counting down)
        assert new_state.pot_cooking_timer[0] == 3

    def test_pot_burns_when_timer_reaches_zero(self):
        """Verify pot contents are cleared when timer hits 0."""
        from jaxmarl.environments.overcooked_v3.common import DynamicObject
        env = OvercookedV3(pot_cook_time=10, pot_burn_time=5)
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)

        # Set timer to 1, so one step brings it to 0 (burned)
        state = self._setup_full_pot(env, state, timer_value=1)

        new_state, key = self._step_noop(env, state, key)

        # Timer should be 0
        assert new_state.pot_cooking_timer[0] == 0

        # Pot should be cleared (ingredients reset to 0)
        pot_y, pot_x = new_state.pot_positions[0]
        pot_ingredients = new_state.grid[pot_y, pot_x, 1]
        assert pot_ingredients == 0

    def test_pot_full_cooking_cycle(self):
        """Test complete cycle: full pot -> cooking -> cooked -> burn window -> burned."""
        from jaxmarl.environments.overcooked_v3.common import DynamicObject
        env = OvercookedV3(pot_cook_time=10, pot_burn_time=5)
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)

        state = self._setup_full_pot(env, state, timer_value=10)
        pot_y, pot_x = state.pot_positions[0]

        # Step through cooking phase (timer 10 -> 6, not yet cooked)
        for expected_timer in range(9, 5, -1):
            state, key = self._step_noop(env, state, key)
            assert state.pot_cooking_timer[0] == expected_timer
            pot_ingredients = state.grid[pot_y, pot_x, 1]
            assert (pot_ingredients & DynamicObject.COOKED) == 0, \
                f"Should not be cooked yet at timer={expected_timer}"

        # Step to burn_time (timer 6 -> 5, becomes cooked)
        state, key = self._step_noop(env, state, key)
        assert state.pot_cooking_timer[0] == 5
        pot_ingredients = state.grid[pot_y, pot_x, 1]
        assert (pot_ingredients & DynamicObject.COOKED) != 0, "Should be cooked at burn_time"

        # Step through burn window (timer 5 -> 1, still cooked)
        for expected_timer in range(4, 0, -1):
            state, key = self._step_noop(env, state, key)
            assert state.pot_cooking_timer[0] == expected_timer
            pot_ingredients = state.grid[pot_y, pot_x, 1]
            assert (pot_ingredients & DynamicObject.COOKED) != 0, \
                f"Should still be cooked at timer={expected_timer}"

        # Final step: timer hits 0, pot burns and clears
        state, key = self._step_noop(env, state, key)
        assert state.pot_cooking_timer[0] == 0
        pot_ingredients = state.grid[pot_y, pot_x, 1]
        assert pot_ingredients == 0, "Pot should be cleared after burning"

    def test_empty_pot_timer_stays_zero(self):
        """Verify timer doesn't change for empty pots."""
        env = OvercookedV3(pot_cook_time=10, pot_burn_time=5)
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)

        # Pot starts empty with timer 0
        assert state.pot_cooking_timer[0] == 0

        new_state, key = self._step_noop(env, state, key)

        # Should stay 0
        assert new_state.pot_cooking_timer[0] == 0


class TestOvercookedV3RewardShaping:
    """Test shaped rewards for plate interactions."""

    def _make_env_and_state(self):
        layout = Layout.from_string(
            """
WWWWWW
WAB XW
W P0 W
W    W
WWWWWW
""",
            possible_recipes=[[0, 0, 0]],
        )
        env = OvercookedV3(layout=layout, pot_cook_time=10, pot_burn_time=5)
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)
        state = state.replace(
            agents=state.agents.replace(
                dir=jnp.array([Direction.RIGHT], dtype=jnp.int32),
            )
        )
        return env, state

    def _set_pot(self, state, contents, timer=0):
        pot_y, pot_x = state.pot_positions[0]
        grid = state.grid.at[pot_y, pot_x, 1].set(jnp.int32(contents))
        timers = state.pot_cooking_timer.at[0].set(jnp.int32(timer))
        return state.replace(grid=grid, pot_cooking_timer=timers)

    def _pickup_plate_reward(self, env, state):
        actions = {"agent_0": int(Actions.interact)}
        _, _, _, _, info = env.step(jax.random.PRNGKey(1), state, actions)
        return float(info["shaped_reward"]["agent_0"])

    @pytest.mark.parametrize("ingredient_count", [1, 2])
    def test_plate_pickup_not_rewarded_for_partial_pot(self, ingredient_count):
        env, state = self._make_env_and_state()
        pot_contents = DynamicObject.ingredient(0) * ingredient_count
        state = self._set_pot(state, pot_contents)

        shaped_reward = self._pickup_plate_reward(env, state)

        assert shaped_reward == pytest.approx(0.0)

    def test_plate_pickup_rewarded_for_full_unburned_pot(self):
        env, state = self._make_env_and_state()
        pot_contents = DynamicObject.ingredient(0) * 3
        state = self._set_pot(state, pot_contents, timer=10)

        shaped_reward = self._pickup_plate_reward(env, state)

        assert shaped_reward == pytest.approx(0.1)

    def test_plate_pickup_not_rewarded_for_burned_pot(self):
        env, state = self._make_env_and_state()
        pot_contents = (DynamicObject.ingredient(0) * 3) | DynamicObject.BURNED
        state = self._set_pot(state, pot_contents)

        shaped_reward = self._pickup_plate_reward(env, state)

        assert shaped_reward == pytest.approx(0.0)


class TestOvercookedV3OrderQueue:
    """Test order queue system."""

    def test_order_queue_disabled_by_default(self):
        """Verify order queue is disabled by default."""
        env = OvercookedV3()
        assert env.enable_order_queue == False

    def test_order_queue_can_be_enabled(self):
        """Verify order queue can be enabled."""
        env = OvercookedV3(enable_order_queue=True)
        assert env.enable_order_queue == True
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)
        assert state.order_types is not None


class TestOvercookedV3Conveyors:
    """Test conveyor belt mechanics."""

    def test_item_conveyors_disabled_by_default(self):
        """Verify item conveyors are disabled by default."""
        env = OvercookedV3()
        assert env.enable_item_conveyors == False

    def test_player_conveyors_disabled_by_default(self):
        """Verify player conveyors are disabled by default."""
        env = OvercookedV3()
        assert env.enable_player_conveyors == False

    def test_conveyor_demo_layout(self):
        """Test conveyor demo layout loads correctly."""
        env = OvercookedV3(layout="conveyor_demo", enable_item_conveyors=True)
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)

        # Should have some item conveyors
        assert jnp.any(state.item_conveyor_active_mask)

    def test_item_conveyor_drops_items_pushed_past_map_border(self):
        """Items pushed out of bounds by an item conveyor should disappear."""
        layout = Layout.from_string(
            """
WWWWWW
WA X W
W0BP >
WWWWWW
""",
            possible_recipes=[[0, 0, 0]],
        )
        env = OvercookedV3(layout=layout, enable_item_conveyors=True)
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)

        item = jnp.int32(DynamicObject.PLATE)
        state = state.replace(grid=state.grid.at[2, 5, 1].set(item))
        assert state.grid[2, 5, 0] == StaticObject.ITEM_CONVEYOR

        actions = {"agent_0": int(Actions.stay)}
        obs, new_state, rewards, dones, info = env.step(
            jax.random.PRNGKey(1), state, actions
        )

        assert new_state.grid[2, 5, 1] == 0

    def test_item_conveyor_keeps_items_pushed_onto_normal_wall(self):
        """In-bounds walls are counters, not border sinks."""
        layout = Layout.from_string(
            """
WWWWWWW
WA X  W
W0BP >W
WWWWWWW
""",
            possible_recipes=[[0, 0, 0]],
        )
        env = OvercookedV3(layout=layout, enable_item_conveyors=True)
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)

        item = jnp.int32(DynamicObject.PLATE)
        state = state.replace(grid=state.grid.at[2, 5, 1].set(item))
        assert state.grid[2, 5, 0] == StaticObject.ITEM_CONVEYOR
        assert state.grid[2, 6, 0] == StaticObject.WALL

        actions = {"agent_0": int(Actions.stay)}
        obs, new_state, rewards, dones, info = env.step(
            jax.random.PRNGKey(1), state, actions
        )

        assert new_state.grid[2, 5, 1] == 0
        assert new_state.grid[2, 6, 1] == item

    def test_agents_cannot_move_onto_item_conveyors(self):
        """Agents should not be able to occupy item conveyor cells."""
        layout = Layout.from_string(
            coordinated_temporal_conveyor,
            possible_recipes=[[0, 0, 0]],
        )
        env = OvercookedV3(
            layout=layout,
            enable_item_conveyors=True,
            enable_player_conveyors=True,
        )
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)

        # Put agent 0 immediately left of the vertical item conveyor at (6, 4).
        state = state.replace(
            agents=state.agents.replace(
                pos=Position(
                    x=jnp.array([5, 2], dtype=jnp.int32),
                    y=jnp.array([4, 4], dtype=jnp.int32),
                )
            )
        )
        assert state.grid[4, 6, 0] == StaticObject.ITEM_CONVEYOR

        actions = {"agent_0": int(Actions.right), "agent_1": int(Actions.stay)}
        key, subkey = jax.random.split(key)
        obs, new_state, rewards, dones, info = env.step(subkey, state, actions)

        assert new_state.agents.pos.x[0] == 5
        assert new_state.agents.pos.y[0] == 4

    def test_agents_can_move_onto_player_conveyors(self):
        """Agents should still be able to occupy player conveyor cells."""
        layout = Layout.from_string(
            """
WWWWWW
WA]X W
W0BP W
WWWWWW
""",
            possible_recipes=[[0, 0, 0]],
        )
        env = OvercookedV3(layout=layout, enable_player_conveyors=False)
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)

        assert state.grid[1, 2, 0] == StaticObject.PLAYER_CONVEYOR

        actions = {"agent_0": int(Actions.right)}
        key, subkey = jax.random.split(key)
        obs, new_state, rewards, dones, info = env.step(subkey, state, actions)

        assert new_state.agents.pos.x[0] == 2
        assert new_state.agents.pos.y[0] == 1

    def test_player_conveyors_do_not_push_agents_onto_item_conveyors(self):
        """Player conveyors should not push agents onto item conveyors."""
        layout = Layout.from_string(
            """
WWWWWW
WA]>XW
W0BP W
WWWWWW
""",
            possible_recipes=[[0, 0, 0]],
        )
        env = OvercookedV3(layout=layout, enable_player_conveyors=True)
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)

        state = state.replace(
            agents=state.agents.replace(
                pos=Position(
                    x=jnp.array([2], dtype=jnp.int32),
                    y=jnp.array([1], dtype=jnp.int32),
                )
            )
        )
        assert state.grid[1, 2, 0] == StaticObject.PLAYER_CONVEYOR
        assert state.grid[1, 3, 0] == StaticObject.ITEM_CONVEYOR

        actions = {"agent_0": int(Actions.stay)}
        key, subkey = jax.random.split(key)
        obs, new_state, rewards, dones, info = env.step(subkey, state, actions)

        assert new_state.agents.pos.x[0] == 2
        assert new_state.agents.pos.y[0] == 1


class TestOvercookedV3Registration:
    """Test environment registration."""

    def test_make_function_works(self):
        """Verify env can be created via jaxmarl.make()."""
        env = make("overcooked_v3")
        assert env is not None
        assert hasattr(env, 'reset')
        assert hasattr(env, 'step')

    def test_make_with_kwargs(self):
        """Verify make works with custom kwargs."""
        env = make("overcooked_v3", max_steps=200)
        assert env.max_steps == 200


class TestOvercookedV3Observations:
    """Test observation generation."""

    def test_observation_shape(self):
        """Verify observation has correct shape."""
        env = OvercookedV3()
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)

        for agent in env.agents:
            assert obs[agent].shape == env.obs_shape

    def test_partial_observability(self):
        """Test partial observability with agent_view_size."""
        env = OvercookedV3(agent_view_size=2)
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)

        # Observation should be smaller than full grid
        assert obs[env.agents[0]].shape[0] <= 5
        assert obs[env.agents[0]].shape[1] <= 5


class TestOvercookedV3Actions:
    """Test action handling."""

    def test_movement_actions(self):
        """Test that movement actions change agent position."""
        env = OvercookedV3()
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)

        # Try moving right
        actions = {env.agents[0]: 0, env.agents[1]: 4}  # right for agent 0, stay for agent 1
        key, subkey = jax.random.split(key)
        obs, new_state, rewards, dones, info = env.step(subkey, state, actions)

        # Direction should have changed to RIGHT regardless of movement
        assert new_state.agents.dir[0] == 2  # Direction.RIGHT

    def test_stay_action(self):
        """Test that stay action doesn't change position."""
        env = OvercookedV3()
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)

        initial_pos = (state.agents.pos.x[0].item(), state.agents.pos.y[0].item())
        initial_dir = state.agents.dir[0].item()

        # Stay action for all agents
        actions = {agent: 4 for agent in env.agents}
        key, subkey = jax.random.split(key)
        obs, new_state, rewards, dones, info = env.step(subkey, state, actions)

        new_pos = (new_state.agents.pos.x[0].item(), new_state.agents.pos.y[0].item())

        assert initial_pos == new_pos
        assert initial_dir == new_state.agents.dir[0].item()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
