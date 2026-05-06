import jax

from jaxmarl.environments.overcooked_v2.layouts import Layout
from jaxmarl.environments.overcooked_v2.overcooked import OvercookedV2


def _small_layout():
    return Layout.from_string(
        """
WWWWW
WA AW
W   W
WX0PW
""",
        possible_recipes=[[0, 0, 0]],
    )


def _goal_channel(env):
    agent_channels = 1 + 4 + env.layout.num_ingredients + 2
    static_layers_start = 2 * agent_channels
    return static_layers_start + 1


def test_agent_view_size_is_not_capped_by_layout_dimensions():
    env = OvercookedV2(layout=_small_layout(), agent_view_size=2)

    obs, _ = env.reset(jax.random.PRNGKey(0))
    agent_obs = obs["agent_0"]

    assert env.obs_shape == (5, 5, 30)
    assert agent_obs.shape == env.obs_shape
    assert agent_obs[4, 2, _goal_channel(env)].item() == 1


def test_none_agent_view_size_uses_full_grid_observation_shape():
    layout = _small_layout()
    env = OvercookedV2(layout=layout, agent_view_size=None)

    obs, _ = env.reset(jax.random.PRNGKey(0))

    assert env.obs_shape == (layout.height, layout.width, 30)
    assert obs["agent_0"].shape == env.obs_shape
