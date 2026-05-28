"""Tests for Layout utility methods."""

import pytest
from jaxmarl.environments.overcooked_v3.layouts import Layout, overcooked_v3_layouts
from jaxmarl.environments.overcooked_v3.overcooked import OvercookedV3


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
