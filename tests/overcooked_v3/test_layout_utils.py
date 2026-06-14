"""Tests for Layout utility methods."""

import numpy as np
import pytest
from jaxmarl.environments.overcooked_v3.common import (
    ButtonAction,
    Direction,
    StaticObject,
)
from jaxmarl.environments.overcooked_v3.layouts import (
    Layout,
    moving_wall_bounce_demo,
    overcooked_v3_layouts,
)
from jaxmarl.environments.overcooked_v3.overcooked import OvercookedV3
from jaxmarl.environments.overcooked_v3.settings import (
    MAX_BARRIERS,
    MAX_BUTTONS,
    MAX_MOVING_WALLS,
)


class TestToString:
    """Test layout to string conversion."""

    def test_round_trip_cramped_room(self):
        """Test that cramped_room can be converted to string and back."""
        original = overcooked_v3_layouts["cramped_room"]

        layout_str = original.to_string()

        reconstructed = Layout.from_string(
            layout_str, possible_recipes=original.possible_recipes
        )

        assert reconstructed.width == original.width
        assert reconstructed.height == original.height

        assert len(reconstructed.agent_positions) == len(original.agent_positions)
        assert set(reconstructed.agent_positions) == set(original.agent_positions)

        assert (reconstructed.static_objects == original.static_objects).all()

    def test_round_trip_conveyor_demo(self):
        """Test that conveyor_demo layout can be converted and reconstructed."""
        original = overcooked_v3_layouts["conveyor_demo"]

        layout_str = original.to_string()
        reconstructed = Layout.from_string(
            layout_str, possible_recipes=original.possible_recipes
        )

        assert len(reconstructed.item_conveyor_info) == len(original.item_conveyor_info)
        assert set(reconstructed.item_conveyor_info) == set(original.item_conveyor_info)

    def test_round_trip_player_conveyor(self):
        """Test that player_conveyor layouts preserve conveyor info."""
        original = overcooked_v3_layouts["player_conveyor_loop"]

        layout_str = original.to_string()
        reconstructed = Layout.from_string(
            layout_str, possible_recipes=original.possible_recipes
        )

        assert len(reconstructed.player_conveyor_info) == len(
            original.player_conveyor_info
        )
        assert set(reconstructed.player_conveyor_info) == set(
            original.player_conveyor_info
        )

    def test_round_trip_all_registered_layouts(self):
        """Test round-trip conversion for all registered layouts."""
        for layout_name, original in overcooked_v3_layouts.items():
            layout_str = original.to_string()
            reconstructed = Layout.from_string(
                layout_str, possible_recipes=original.possible_recipes
            )

            assert reconstructed.width == original.width, f"Failed for {layout_name}"
            assert reconstructed.height == original.height, f"Failed for {layout_name}"
            assert len(reconstructed.agent_positions) == len(
                original.agent_positions
            ), f"Failed for {layout_name}"
            assert (reconstructed.static_objects == original.static_objects).all(), (
                f"Failed for {layout_name}"
            )


class TestGetInfo:
    """Test layout information extraction."""

    def test_cramped_room_info(self):
        """Test info extraction for cramped_room."""
        layout = overcooked_v3_layouts["cramped_room"]
        info = layout.get_info()

        assert info["dimensions"] == (layout.width, layout.height)
        assert info["num_agents"] == 2
        assert info["num_pots"] == 1
        assert info["num_goals"] == 1
        assert info["num_plate_piles"] == 1
        assert 0 in info["num_ingredient_piles"]

    def test_conveyor_demo_info(self):
        """Test info extraction for layout with conveyors."""
        layout = overcooked_v3_layouts["conveyor_demo"]
        info = layout.get_info()

        assert info["num_item_conveyors"] > 0

    def test_player_conveyor_info(self):
        """Test info extraction for player conveyor layout."""
        layout = overcooked_v3_layouts["player_conveyor_loop"]
        info = layout.get_info()

        assert info["num_player_conveyors"] > 0


class TestValidate:
    """Test layout validation."""

    def test_valid_layout(self):
        """Test that registered layouts pass validation."""
        layout = overcooked_v3_layouts["cramped_room"]
        is_valid, messages = layout.validate()

        assert is_valid, f"cramped_room should be valid, got: {messages}"

    def test_missing_agents(self):
        """Test that layout without agents is rejected at construction."""
        layout_str = """
WWPWW
0   0
W   W
WBWXW
"""
        with pytest.raises(
            ValueError, match="At least one agent position must be provided"
        ):
            Layout.from_string(layout_str, possible_recipes=[[0, 0, 0]])

    def test_missing_goal(self):
        """Test that layout without delivery zone fails validation."""
        layout_str = """
WWPWW
0A A0
W   W
WBWWW
"""
        layout = Layout.from_string(layout_str, possible_recipes=[[0, 0, 0]])
        is_valid, messages = layout.validate()

        assert not is_valid
        assert any(
            "goal" in msg.lower() or "delivery" in msg.lower() for msg in messages
        )

    def test_missing_ingredients(self):
        """Test that layout without ingredients fails validation."""
        layout_str = """
WWPWW
WA AW
W   W
WBWXW
"""
        layout = Layout.from_string(layout_str, possible_recipes=[[0, 0, 0]])
        is_valid, messages = layout.validate()

        assert not is_valid
        assert any("ingredient" in msg.lower() for msg in messages)

    def test_invalid_moving_wall_button_target_fails_validation(self):
        """Buttons cannot target moving wall indexes that do not exist."""
        layout_str = """
WWWWWW
W0A eW
W P !X
W B  W
WWWWWW
"""
        layout = Layout.from_string(
            layout_str,
            possible_recipes=[[0, 0, 0]],
            button_config=[(2, ButtonAction.TOGGLE_PAUSE)],
        )

        is_valid, messages = layout.validate()

        assert not is_valid
        assert any("moving wall 2" in msg.lower() for msg in messages)

    def test_multi_target_moving_wall_button_config_validates(self):
        """A button can target more than one moving wall."""
        layout = Layout.from_string(
            moving_wall_bounce_demo,
            possible_recipes=[[0, 0, 0]],
            moving_wall_bounce=[True, True],
            button_config=[([0, 1], ButtonAction.TOGGLE_PAUSE)],
        )

        is_valid, messages = layout.validate()
        env = OvercookedV3(layout=layout)

        assert is_valid, messages
        assert layout.button_info[0][2] == (0, 1)
        assert env._button_target_idxs[0, 0] == 0
        assert env._button_target_idxs[0, 1] == 1
        assert env._button_target_mask[0, 0]
        assert env._button_target_mask[0, 1]

    def test_invalid_barrier_button_target_fails_validation(self):
        """Buttons cannot target barrier indexes that do not exist."""
        layout_str = """
WWWWWW
W0A #W
W P !X
W B  W
WWWWWW
"""
        layout = Layout.from_string(
            layout_str,
            possible_recipes=[[0, 0, 0]],
            button_config=[(1, ButtonAction.TIMED_BARRIER)],
            barrier_config=[True],
        )

        is_valid, messages = layout.validate()

        assert not is_valid
        assert any("barrier 1" in msg.lower() for msg in messages)

    def test_invalid_button_action_type_fails_validation(self):
        """Button action types must be valid ButtonAction values."""
        layout_str = """
WWWWWW
W0A eW
W P !X
W B  W
WWWWWW
"""
        layout = Layout.from_string(
            layout_str,
            possible_recipes=[[0, 0, 0]],
            button_config=[(0, 999)],
        )

        is_valid, messages = layout.validate()

        assert not is_valid
        assert any("invalid action type" in msg.lower() for msg in messages)

    def test_button_action_wrong_target_family_fails_validation(self):
        """Barrier actions must target barriers, not moving walls."""
        layout_str = """
WWWWWW
W0A eW
W P !X
W B  W
WWWWWW
"""
        layout = Layout.from_string(
            layout_str,
            possible_recipes=[[0, 0, 0]],
            button_config=[(0, ButtonAction.TIMED_BARRIER)],
        )

        is_valid, messages = layout.validate()

        assert not is_valid
        assert any("targets barrier 0" in msg.lower() for msg in messages)

    def test_validate_rejects_too_many_dynamic_mechanics(self):
        """Validation catches layouts that exceed fixed JAX state capacities."""
        width = MAX_MOVING_WALLS + MAX_BUTTONS + MAX_BARRIERS + 4
        static_objects = np.full((3, width), StaticObject.EMPTY, dtype=int)
        static_objects[1, 1] = StaticObject.GOAL
        static_objects[1, 2] = StaticObject.POT
        static_objects[1, 3] = StaticObject.PLATE_PILE
        static_objects[2, 1] = StaticObject.INGREDIENT_PILE_BASE

        moving_wall_info = []
        for idx in range(MAX_MOVING_WALLS + 1):
            static_objects[0, idx] = StaticObject.MOVING_WALL
            moving_wall_info.append((0, idx, Direction.RIGHT, False))

        button_info = []
        button_offset = MAX_MOVING_WALLS + 1
        for idx in range(MAX_BUTTONS + 1):
            x = button_offset + idx
            static_objects[0, x] = StaticObject.BUTTON
            button_info.append((0, x, 0, ButtonAction.TOGGLE_DIRECTION))

        barrier_info = []
        barrier_offset = button_offset + MAX_BUTTONS + 1
        for idx in range(MAX_BARRIERS + 1):
            x = barrier_offset + idx
            static_objects[0, x] = StaticObject.BARRIER
            barrier_info.append((0, x, True))

        layout = Layout(
            agent_positions=[(1, 0)],
            static_objects=static_objects,
            num_ingredients=1,
            possible_recipes=[[0, 0, 0]],
            moving_wall_info=moving_wall_info,
            button_info=button_info,
            barrier_info=barrier_info,
        )

        is_valid, messages = layout.validate()

        assert not is_valid
        assert any("too many moving walls" in msg.lower() for msg in messages)
        assert any("too many buttons" in msg.lower() for msg in messages)
        assert any("too many barriers" in msg.lower() for msg in messages)

    def test_validate_rejects_bad_recipe_shapes(self):
        """Recipe validation lives in validate(), not only in from_string()."""
        layout_str = """
WWWWW
W0APW
WBX W
WWWWW
"""
        layout = Layout.from_string(layout_str, possible_recipes=[[0, 0]])
        is_valid, messages = layout.validate()

        assert not is_valid
        assert any("recipe 0" in msg.lower() for msg in messages)

    def test_moving_wall_bounce_demo_validates(self):
        """Registered moving wall bounce demo has valid button target indexes."""
        layout = overcooked_v3_layouts["moving_wall_bounce_demo"]

        is_valid, validate_messages = layout.validate()
        is_playable, playable_messages = layout.validate_playable()

        assert is_valid, validate_messages
        assert is_playable, playable_messages

    def test_ragged_layout_rejected(self):
        """Test that implicit empty padding is rejected."""
        layout_str = """
WWWWW
W0A
WWWWW
"""
        with pytest.raises(ValueError, match="rectangular"):
            Layout.from_string(layout_str, possible_recipes=[[0, 0, 0]])

    def test_playable_validation_requires_plate(self):
        """Test that soup-delivery validation treats missing plates as fatal."""
        layout_str = """
WWWWW
W0APW
WWXWW
"""
        layout = Layout.from_string(layout_str, possible_recipes=[[0, 0, 0]])
        is_playable, messages = layout.validate_playable()

        assert not is_playable
        assert any("plate" in msg.lower() for msg in messages)

    def test_playable_validation_rejects_mixed_recipe(self):
        """Test current pot mechanics reject mixed recipes."""
        layout_str = """
WWWWWWWW
W0A1   W
W P B XW
WWWWWWWW
"""
        layout = Layout.from_string(layout_str, possible_recipes=[[0, 0, 1]])
        is_playable, messages = layout.validate_playable()

        assert not is_playable
        assert any("mixed" in msg.lower() for msg in messages)

    def test_env_init_rejects_unplayable_layout(self):
        """Test OvercookedV3 raises immediately for unplayable layouts."""
        layout_str = """
WWWWW
W0APW
WWXWW
"""
        layout = Layout.from_string(layout_str, possible_recipes=[[0, 0, 0]])

        with pytest.raises(ValueError, match="Invalid OvercookedV3 layout"):
            OvercookedV3(layout=layout)


class TestAnnotateLayoutString:
    """Test layout annotation."""

    def test_annotation_includes_legend(self):
        """Test that annotation adds a legend."""
        layout_str = """
WWPWW
0A A0
W   W
WBWXW
"""
        annotated = Layout.annotate_layout_string(layout_str)

        assert "Symbol Legend" in annotated
        assert "W = Wall" in annotated
        assert "P = Pot" in annotated
        assert layout_str in annotated
