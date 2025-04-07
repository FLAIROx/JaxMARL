"""
Test the environments to ensure they implement the base API correctly.
STORM is not included as its observation space does not follow the base API.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp

# Import the base environment class
from jaxmarl.environments import MultiAgentEnv
from jaxmarl.environments.spaces import Space
# Import the specific environment to test. Replace 'MyEnv' with your environment's class.

from jaxmarl.registration import make

envs_to_test = [
    "MPE_simple_v3",
    "MPE_simple_tag_v3",
    "MPE_simple_world_comm_v3",
    "MPE_simple_spread_v3",
    "MPE_simple_crypto_v3",
    "MPE_simple_speaker_listener_v4",
    "MPE_simple_push_v3",
    "MPE_simple_adversary_v3",
    "MPE_simple_reference_v3",
    "MPE_simple_facmac_v1",
    "MPE_simple_facmac_3a_v1",
    "MPE_simple_facmac_6a_v1",
    "MPE_simple_facmac_9a_v1",
    "switch_riddle",
    "SMAX",
    "HeuristicEnemySMAX",
    "ant_4x2",
    "halfcheetah_6x1",
    "hopper_3x1",
    "humanoid_9|8",
    "walker2d_2x3",
    # "storm",
    # "storm_2p",
    # "storm_np",
    "hanabi",
    "overcooked",
    "overcooked_v2",
    "coin_game",
    "jaxnav",
]

# A pytest fixture to instantiate the environment.
@pytest.fixture(scope="module", params=envs_to_test)
def env(request):
    env_id = request.param
    env_instance = make(env_id)
    yield env_instance

def test_inherits_base_env(env: MultiAgentEnv):
    """Test that the environment is a subclass of MultiAgentEnv."""
    
    assert isinstance(env, MultiAgentEnv), "Environment does not inherit from MultiAgentEnv"

def test_observation_space_definition(env: MultiAgentEnv):
    assert hasattr(env, "observation_spaces"), "`observation_spaces` does  not exist"

    for agent in env.observation_spaces.keys():
        assert env.observation_spaces[agent] == env.observation_space(agent)
        assert isinstance(env.observation_spaces[agent], Space), f"Agent observation space {env.observation_spaces[agent]} is not a valid JaxMARL space"

def test_action_space_definition(env: MultiAgentEnv):
    assert hasattr(env, "action_spaces"), "`action_spaces` does  not exist"

    for agent in env.action_spaces.keys():
        assert env.action_spaces[agent] == env.action_space(agent)
        assert isinstance(env.action_spaces[agent], Space)

def test_reset_returns_valid_observation(env: MultiAgentEnv):
    """Test that reset() returns an observation that is valid according to observation_space."""
    # Check that the environment defines an observation_space attribute.
    rng = jax.random.PRNGKey(0)

    initial_obs, _ = env.reset(rng)
    # Verify that the returned observation is contained in the observation space.
    for agent in env.observation_spaces.keys():
        print(f"Agent: {agent}, Observation: {initial_obs[agent]}, Space: {env.observation_spaces[agent]}")
        assert env.observation_spaces[agent].contains(initial_obs[agent]), f"Initial observation for agent {agent}, {env.observation_spaces[agent]} does not contain the observation value {initial_obs[agent]}"


def test_step_returns_correct_format(env):
    """Test that step(action) returns a tuple (observation, reward, done, info)
       with valid types and that the observation adheres to the observation_space."""
    # Ensure the environment defines an action_space attribute.
    assert hasattr(env, "action_space"), "Environment missing action_space attribute"
    
    rng = jax.random.PRNGKey(0)

    # Reset the environment first.
    rng, _rng = jax.random.split(rng)
    _, initial_state = env.reset(_rng)
    # Sample a valid action.

    rng, _rng = jax.random.split(rng)
    actions = {
        a: env.action_space(a).sample(_rng) for a in env.agents
    }
    
    # Take a step in the environment.
    rng, _rng = jax.random.split(rng)
    result = env.step(_rng, initial_state, actions)
    # Check that the result is a 4-tuple.
    assert isinstance(result, tuple) and len(result) == 5, "Step did not return a 5-tuple"
    
    next_obs, next_state, reward, done, info = result
    for agent in env.observation_spaces.keys():
        assert env.observation_spaces[agent].contains(next_obs[agent])

    # Check that the reward is numeric.
    assert isinstance(reward, dict), "Reward is not a dictionary"
    # Verify that 'done' is a boolean.
    assert isinstance(done, dict), "Done flag is not a dictionary"
    assert "__all__" in done

    # Check that 'info' is a dictionary.
    assert isinstance(info, dict), "Info is not a dictionary"

@pytest.mark.parametrize("env_id", envs_to_test)
def test_envs(env_id):
    env = make(env_id)
    test_inherits_base_env(env)
    test_observation_space_definition(env)
    test_action_space_definition(env)
    test_reset_returns_valid_observation(env)
    test_step_returns_correct_format(env)

if __name__ == "__main__":
    for env_id in envs_to_test:
        env_instance = make(env_id)
        print(f"Testing environment: {env_id}")
        test_inherits_base_env(env_instance)
        test_observation_space_definition(env_instance)
        test_action_space_definition(env_instance)
        test_reset_returns_valid_observation(env_instance)
        test_step_returns_correct_format(env_instance)
