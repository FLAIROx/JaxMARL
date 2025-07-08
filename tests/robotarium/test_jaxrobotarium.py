import pytest
import jax
import jax.numpy as jnp

from jaxrobotarium.robotarium_env import *

@pytest.fixture
def env():
    num_agents = 3
    env = RobotariumEnv(num_agents=num_agents, action_type="Continuous")
    key = jax.random.PRNGKey(0)
    return env, num_agents

def test_get_violations(env):
    r_env, num_agents = env
    state = State(
        p_pos = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        done = jnp.full((num_agents,), False),
        step = 0
    )
    violations = r_env._get_violations(state)
    assert violations['collision'] == 3

def test_observation_space(env):
    r_env, num_agents = env
    r_env.observation_spaces = {str(i): "obs_space" for i in range(num_agents)}
    for i in range(num_agents):
        assert r_env.observation_space(str(i)) == "obs_space"

def test_action_space(env):
    r_env, num_agents = env
    r_env.action_spaces = {str(i): "act_space" for i in range(num_agents)}
    for i in range(num_agents):
        assert r_env.action_space(str(i)) == "act_space"

def test_discrete_action_decoder(env):
    r_env, num_agents = env
    state = State(
        p_pos = jnp.array([[0.0, 0.9, 0.0], [0.0, -0.9, 0.0], [1.5, 0.0, 0.0]]),
        done = jnp.full((num_agents,), False),
        step = 0
    )
    decoded_actions = [r_env._decode_discrete_action(i, i+1, state) for i in range(num_agents)]
    for i in range(num_agents):
        assert -1.6 <= decoded_actions[i][0] <= 1.6
        assert -1 <= decoded_actions[i][1] <= 1

    state = State(
        p_pos = jnp.array([[0.0, 1.1, 0.0], [0.0, -1.1, 0.0], [1.7, 0.0, 0.0]]),
        done = jnp.full((num_agents,), False),
        step = 0
    )
    decoded_actions = [r_env._decode_discrete_action(i, i+1, state) for i in range(num_agents)]
    for i in range(num_agents):
        assert -1.6 <= decoded_actions[i][0] <= 1.6
        assert -1 <= decoded_actions[i][1] <= 1

def test_continuous_action_decoder(env):
    r_env, num_agents = env
    state = State(
        p_pos = jnp.array([[0.0, 0.9, 0.0], [0.0, -0.9, 0.0], [1.5, 0.0, 0.0]]),
        done = jnp.full((num_agents,), False),
        step = 0
    )
    actions = jnp.array([[0, 0], [0.1, 0.1], [0.2, 0.2]])
    decoded_actions = [r_env._decode_continuous_action(i, actions[i], state) for i in range(num_agents)]
    for i in range(num_agents):
        assert decoded_actions[i][0] == actions[i][0]
        assert decoded_actions[i][1] == actions[i][1]

def test_robotarium_step():
    num_agents = 3
    poses = jnp.array([[0., 0, 0]])
    goals = jnp.array([[1., 0]])
    r_env = RobotariumEnv(
        num_agents=num_agents,
        action_type="Discrete",
        controller={"controller": "clf_uni_position"},
        update_frequency=30
    )
    final_pose = r_env._robotarium_step(poses, goals)
    assert jnp.linalg.norm(poses - final_pose) > 0.1
