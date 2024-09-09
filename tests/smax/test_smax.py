import jax.numpy as jnp
import jax
from jaxmarl import make
from jaxmarl.environments.smax.smax_env import State
import pytest


def create_env(key, continuous_action=False, conic_observation=False):
    env = make(
        "SMAX",
        num_allies=5,
        num_enemies=5,
        map_width=32,
        map_height=32,
        world_steps_per_env_step=8,
        unit_type_velocities=jnp.array([5.0]),
        unit_type_health=jnp.array([1.0]),
        unit_type_attacks=jnp.array([0.02]),
        unit_type_weapon_cooldowns=jnp.array([0.2]),
        time_per_step=1.0 / 16,
        won_battle_bonus=5.0,
        unit_type_attack_ranges=jnp.array([3.0]),
        unit_type_sight_ranges=jnp.array([4.0]),
        max_steps=100,
        action_type="discrete" if not continuous_action else "continuous",
        observation_type="unit_list" if not conic_observation else "conic",
    )
    obs, state = env.reset(key)
    return env, obs, state


def get_random_actions(key, env):
    key_a = jax.random.split(key, num=env.num_agents)
    return {
        agent: env.action_space(agent).sample(key_a[i])
        for i, agent in enumerate(env.agents)
    }


@pytest.mark.parametrize(
    ("action", "vec_diff", "do_jit"),
    [
        (0, jnp.array([0, 2.5]), False),
        (0, jnp.array([0, 2.5]), True),
        (1, jnp.array([2.5, 0]), False),
        (1, jnp.array([2.5, 0]), True),
        (2, jnp.array([0, -2.5]), False),
        (2, jnp.array([0, -2.5]), True),
        (3, jnp.array([-2.5, 0]), False),
        (3, jnp.array([-2.5, 0]), True),
    ],
)
def test_move_actions(action, vec_diff, do_jit):
    with jax.disable_jit(do_jit):
        key = jax.random.PRNGKey(0)
        key, key_reset = jax.random.split(key)
        env, _, state = create_env(key_reset)

        init_pos = jnp.array([16, 16])
        unit_positions = state.unit_positions.at[0].set(init_pos)
        state = state.replace(unit_positions=unit_positions)
        key, key_action = jax.random.split(key)
        actions = get_random_actions(key_action, env)
        actions["ally_0"] = action
        key, key_step = jax.random.split(key)
        _, state, _, _, _ = env.step(key_step, state, actions)
        assert jnp.allclose(state.unit_positions[0], init_pos + vec_diff)


@pytest.mark.parametrize(
    ("action", "vec_diff", "do_jit"),
    [
        (jnp.array([0.0, 1.0, 0.5, 0.0]), jnp.array([0.5, 0.0]), True),
        (jnp.array([0.0, 1.0, 0.5, 0.0]), jnp.array([0.5, 0.0]), False),
        (
            jnp.array([0.0, 1.0, 1.0, 0.125]),
            jnp.array([1.0 / jnp.sqrt(2), 1.0 / jnp.sqrt(2)]),
            True,
        ),
        (
            jnp.array([0.0, 1.0, 1.0, 0.125]),
            jnp.array([1.0 / jnp.sqrt(2), 1.0 / jnp.sqrt(2)]),
            False,
        ),
        (jnp.array([0.0, 0.0, 1.0, 0.125]), jnp.array([0.0, 0.0]), True),
        (jnp.array([0.0, 0.0, 1.0, 0.125]), jnp.array([0.0, 0.0]), False),
    ],
)
def test_continuous_move_action_decoding(action, vec_diff, do_jit):
    with jax.disable_jit(do_jit):
        key = jax.random.PRNGKey(0)
        key, key_reset = jax.random.split(key)
        env, _, state = create_env(key_reset, continuous_action=True)
        init_pos = jnp.array([16, 16])
        unit_positions = state.unit_positions.at[0].set(init_pos)
        state = state.replace(unit_positions=unit_positions)
        key, key_action = jax.random.split(key)
        actions = get_random_actions(key_action, env)
        actions["ally_0"] = action
        actions = jnp.array([actions[i] for i in env.agents])
        movement_actions, _ = env._decode_actions(key, state, actions)
        assert jnp.allclose(movement_actions[0], vec_diff)


@pytest.mark.parametrize(
    (
        "unit_1_idx",
        "unit_2_idx",
        "unit_1_pos",
        "unit_2_pos",
        "unit_1_action",
        "unit_2_action",
        "unit_1_expected_action",
        "unit_2_expected_action",
        "do_jit",
    ),
    [
        (
            0,
            5,
            jnp.array([1.0, 1.0]),
            jnp.array([1.0, 2.0]),
            jnp.array([1.0, 0.0, 0.5, 0.25]),
            jnp.array([1.0, 0.0, 0.5, 0.75]),
            5,
            9,
            False,
        ),
        (
            0,
            5,
            jnp.array([1.0, 1.0]),
            jnp.array([1.0, 2.0]),
            jnp.array([1.0, 0.0, 0.5, 0.25]),
            jnp.array([1.0, 0.0, 0.5, 0.75]),
            5,
            9,
            True,
        ),
        (
            0,
            5,
            jnp.array([1.0, 1.0]),
            jnp.array([1.0, 8.0]),
            jnp.array([1.0, 0.0, 0.5, 0.25]),
            jnp.array([1.0, 0.0, 1.0, 0.75]),
            5,  # still want to decode to an attack action even when
            9,  # because the world_step function should handle this case
            False,
        ),
        (
            0,
            5,
            jnp.array([1.0, 1.0]),
            jnp.array([1.0, 8.0]),
            jnp.array([1.0, 0.0, 0.5, 0.25]),
            jnp.array([1.0, 0.0, 1.0, 0.75]),
            5,  # still want to decode to an attack action even when
            9,  # because the world_step function should handle this case
            True,
        ),
    ],
)
def test_continuous_attack_action_decoding(
    unit_1_idx,
    unit_2_idx,
    unit_1_pos,
    unit_2_pos,
    unit_1_action,
    unit_2_action,
    unit_1_expected_action,
    unit_2_expected_action,
    do_jit,
):
    with jax.disable_jit(do_jit):
        key = jax.random.PRNGKey(0)
        key, key_reset = jax.random.split(key)
        env, _, state = create_env(key_reset, continuous_action=True)

        unit_positions = state.unit_positions.at[unit_1_idx].set(unit_1_pos)
        unit_positions = unit_positions.at[unit_2_idx].set(unit_2_pos)
        state = state.replace(unit_positions=unit_positions)

        key, key_actions = jax.random.split(key)
        actions = get_random_actions(key_actions, env)
        actions[f"ally_{unit_1_idx}"] = unit_1_action
        actions[f"enemy_{unit_2_idx - env.num_allies}"] = unit_2_action
        actions = jnp.array([actions[i] for i in env.agents])
        _, attack_actions = env._decode_actions(key, state, actions)
        assert attack_actions[unit_1_idx] == unit_1_expected_action
        assert attack_actions[unit_2_idx] == unit_2_expected_action


@pytest.mark.parametrize(
    (
        "unit_1_idx",
        "unit_2_idx",
        "unit_1_pos",
        "unit_2_pos",
        "unit_1_action",
        "unit_1_prev_action",
        "unit_2_action",
        "unit_2_prev_action",
        "unit_1_expected_action",
        "unit_2_expected_action",
        "do_jit",
    ),
    [
        (
            0,
            5,
            jnp.array([1.0, 1.0]),
            jnp.array([1.0, 2.0]),
            jnp.array([1.0, 0.0, 0.5, 0.25]),
            6,
            jnp.array([1.0, 0.0, 0.5, 0.75]),
            6,
            5,
            9,
            False,
        ),
        (
            0,
            5,
            jnp.array([1.0, 1.0]),
            jnp.array([1.0, 2.0]),
            jnp.array([1.0, 0.0, 0.5, 0.25]),
            6,
            jnp.array([1.0, 0.0, 0.5, 0.75]),
            6,
            5,
            9,
            True,
        ),
        (
            0,
            5,
            jnp.array([1.0, 1.0]),
            jnp.array([1.0, 2.0]),
            jnp.array([0.0, 0.0, 0.5, 0.25]),
            6,
            jnp.array([0.0, 0.0, 0.5, 0.75]),
            6,
            6,
            6,
            False,
        ),
        (
            0,
            5,
            jnp.array([1.0, 1.0]),
            jnp.array([1.0, 2.0]),
            jnp.array([0.0, 0.0, 0.5, 0.25]),
            6,
            jnp.array([0.0, 0.0, 0.5, 0.75]),
            6,
            6,
            6,
            False,
        ),
    ],
)
def test_attack_same_unit(
    unit_1_idx,
    unit_2_idx,
    unit_1_pos,
    unit_2_pos,
    unit_1_action,
    unit_1_prev_action,
    unit_2_action,
    unit_2_prev_action,
    unit_1_expected_action,
    unit_2_expected_action,
    do_jit,
):
    with jax.disable_jit(do_jit):
        key = jax.random.PRNGKey(0)
        key, key_reset = jax.random.split(key)
        env, _, state = create_env(key_reset, continuous_action=True)

        unit_positions = state.unit_positions.at[unit_1_idx].set(unit_1_pos)
        unit_positions = unit_positions.at[unit_2_idx].set(unit_2_pos)
        state = state.replace(unit_positions=unit_positions)

        prev_actions = state.prev_attack_actions.at[unit_1_idx].set(unit_1_prev_action)
        prev_actions = prev_actions.at[unit_2_idx].set(unit_2_prev_action)
        state = state.replace(prev_attack_actions=prev_actions)

        key, key_actions = jax.random.split(key)
        actions = get_random_actions(key_actions, env)
        actions[f"ally_{unit_1_idx}"] = unit_1_action
        actions[f"enemy_{unit_2_idx - env.num_allies}"] = unit_2_action
        actions = jnp.array([actions[i] for i in env.agents])
        _, attack_actions = env._decode_actions(key, state, actions)
        assert attack_actions[unit_1_idx] == unit_1_expected_action
        assert attack_actions[unit_2_idx] == unit_2_expected_action


@pytest.mark.parametrize(
    (
        "unit_1_idx",
        "unit_2_idx",
        "unit_1_pos",
        "unit_2_pos",
        "unit_1_health",
        "unit_2_health",
        "unit_1_reward",
        "unit_2_reward",
        "do_jit",
    ),
    [
        (
            0,
            5,
            jnp.array([1.0, 1.0]),
            jnp.array([1.0, 2.0]),
            0.96,
            0.96,
            0.008,
            0.008,
            False,
        ),
        (
            0,
            5,
            jnp.array([1.0, 1.0]),
            jnp.array([1.0, 2.0]),
            0.96,
            0.96,
            0.008,
            0.008,
            True,
        ),
        (
            0,
            5,
            jnp.array([1.0, 1.0]),
            jnp.array([30.0, 30.0]),
            1.0,
            1.0,
            0.0,
            0.0,
            False,
        ),
        (
            0,
            5,
            jnp.array([1.0, 1.0]),
            jnp.array([30.0, 30.0]),
            1.0,
            1.0,
            0.0,
            0.0,
            True,
        ),
    ],
)
def test_attack_actions(
    unit_1_idx,
    unit_2_idx,
    unit_1_pos,
    unit_2_pos,
    unit_1_health,
    unit_2_health,
    unit_1_reward,
    unit_2_reward,
    do_jit,
):
    with jax.disable_jit(do_jit):
        key = jax.random.PRNGKey(0)
        key, key_reset = jax.random.split(key)
        env, _, state = create_env(key_reset)

        unit_1_action = unit_2_idx - env.num_enemies + env.num_movement_actions
        unit_2_action = env.num_allies - unit_1_idx + env.num_movement_actions - 1

        unit_positions = state.unit_positions.at[unit_1_idx].set(unit_1_pos)
        unit_positions = unit_positions.at[unit_2_idx].set(unit_2_pos)
        state = state.replace(unit_positions=unit_positions)

        key, key_actions = jax.random.split(key)
        actions = get_random_actions(key_actions, env)
        actions[f"ally_{unit_1_idx}"] = unit_1_action
        actions[f"enemy_{unit_2_idx - env.num_allies}"] = unit_2_action

        key, key_step = jax.random.split(key)
        _, state, rewards, _, _ = env.step(key_step, state, actions)

        assert jnp.allclose(state.unit_health[unit_1_idx], unit_1_health)
        assert jnp.allclose(state.unit_health[unit_2_idx], unit_2_health)
        assert jnp.allclose(rewards[f"ally_{unit_1_idx}"], unit_1_reward)
        assert jnp.allclose(
            rewards[f"enemy_{unit_2_idx - env.num_allies}"], unit_2_reward
        )


@pytest.mark.parametrize(
    ("ally_health", "enemy_health", "done_unit", "reward_unit", "do_jit"),
    [
        (1.0, 0.04, "enemy_0", "ally_0", True),
        (1.0, 0.04, "enemy_0", "ally_0", False),
        (0.04, 1.0, "ally_0", "enemy_0", True),
        (0.04, 1.0, "ally_0", "enemy_0", False),
    ],
)
def test_episode_end(ally_health, enemy_health, done_unit, reward_unit, do_jit):
    with jax.disable_jit(do_jit):
        key = jax.random.PRNGKey(0)
        key, key_reset = jax.random.split(key)
        env, _, state = create_env(key_reset)

        unit_1_idx = 0
        unit_2_idx = env.num_allies

        unit_1_health = ally_health
        unit_2_health = enemy_health

        unit_1_action = unit_2_idx - env.num_enemies + env.num_movement_actions
        unit_2_action = env.num_allies - unit_1_idx + env.num_movement_actions - 1

        unit_positions = state.unit_positions.at[unit_1_idx].set(jnp.array([1.0, 1.0]))
        unit_positions = unit_positions.at[unit_2_idx].set(jnp.array([1.0, 2.0]))
        unit_alive = jnp.zeros((env.num_agents,), dtype=jnp.bool_)
        unit_alive = unit_alive.at[unit_1_idx].set(1)
        unit_alive = unit_alive.at[unit_2_idx].set(1)

        unit_health = jnp.zeros((env.num_agents,))
        unit_health = unit_health.at[unit_1_idx].set(unit_1_health)
        unit_health = unit_health.at[unit_2_idx].set(unit_2_health)

        state = state.replace(
            unit_positions=unit_positions,
            unit_alive=unit_alive,
            unit_health=unit_health,
        )

        key, key_actions = jax.random.split(key)
        actions = get_random_actions(key_actions, env)
        actions[f"ally_{unit_1_idx}"] = unit_1_action
        actions[f"enemy_{unit_2_idx - env.num_allies}"] = unit_2_action

        key, key_step = jax.random.split(key)
        _, state, rewards, dones, _ = env.step(key_step, state, actions)
        assert dones[done_unit]
        assert dones["__all__"]
        assert jnp.allclose(
            rewards[reward_unit],
            env.won_battle_bonus
            + 0.04
            / (jnp.sum(env.unit_type_health[state.unit_types[env.num_allies :]])),
        )


@pytest.mark.parametrize(("do_jit"), [True, False])
def test_episode_time_limit(do_jit):
    with jax.disable_jit(do_jit):
        key = jax.random.PRNGKey(0)
        key, key_reset = jax.random.split(key)
        env, _, state = create_env(key_reset)
        state = state.replace(time=env.max_steps)

        key, key_actions = jax.random.split(key)
        actions = get_random_actions(key_actions, env)

        key, key_step = jax.random.split(key)

        _, state, _, dones, _ = env.step(key_step, state, actions)

        assert dones["__all__"]


@pytest.mark.parametrize(("do_jit"), [False, True])
def test_continuous_obs_function(do_jit):
    with jax.disable_jit(do_jit):
        key = jax.random.PRNGKey(0)
        key, key_reset = jax.random.split(key)
        env, _, state = create_env(key_reset, conic_observation=True)
        unit_positions = state.unit_positions.at[0].set(jnp.array([1.0, 1.0]))
        unit_positions = unit_positions.at[1].set(jnp.array([1.55, 1.5]))
        # sweep around from -pi to pi
        expected_obs_idx = 19
        state = state.replace(unit_positions=unit_positions)
        obs = env.get_obs(state)
        real_obs = obs["ally_0"][
            expected_obs_idx
            * 2
            * len(env.unit_features) : (expected_obs_idx * 2 + 1)
            * len(env.unit_features)
        ]
        assert jnp.allclose(
            real_obs,
            jnp.array([1.0, 0.1374999999, 0.125, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
        )


@pytest.mark.parametrize(("do_jit"), [False, True])
def test_continuous_obs_max_two_observed(do_jit):
    with jax.disable_jit(do_jit):
        key = jax.random.PRNGKey(0)
        key, key_reset = jax.random.split(key)
        env, _, state = create_env(key_reset, conic_observation=True)
        unit_positions = state.unit_positions.at[0].set(jnp.array([1.0, 1.0]))
        unit_positions = unit_positions.at[1:4].set(jnp.array([1.55, 1.5]))
        # sweep around from -pi to pi
        expected_obs_idx = 19
        state = state.replace(unit_positions=unit_positions)
        obs = env.get_obs(state)
        real_obs = obs["ally_0"][
            expected_obs_idx
            * 2
            * len(env.unit_features) : (expected_obs_idx * 2 + 1)
            * len(env.unit_features)
        ]
        expected_obs = jnp.array([1.0, 0.1374999999, 0.125, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        assert jnp.allclose(
            real_obs,
            expected_obs,
        )

        real_obs = obs["ally_0"][
            (expected_obs_idx * 2 + 1) * len(env.unit_features):
            (2 * expected_obs_idx + 2) * len(env.unit_features)
        ]
        assert jnp.allclose(real_obs, expected_obs)

        real_obs = obs["ally_0"][
            (expected_obs_idx * 2 + 2) * len(env.unit_features):
            (expected_obs_idx * 2 + 3) * len(env.unit_features)
        ]
        assert jnp.allclose(real_obs, jnp.zeros_like(real_obs))

@pytest.mark.parametrize(("do_jit"), [False, True])
def test_obs_function(do_jit):
    with jax.disable_jit(do_jit):
        key = jax.random.PRNGKey(0)
        key, key_reset = jax.random.split(key)
        env, obs, state = create_env(key_reset)
        first_enemy_idx = (env.num_allies - 1) * len(env.unit_features)
        assert jnp.allclose(
            obs["ally_0"][0 : len(env.unit_features)],
            jnp.array([1.0, -0.5755913, 0.43648314, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
        )
        assert jnp.allclose(
            obs["ally_0"][first_enemy_idx : first_enemy_idx + len(env.unit_features)],
            jnp.zeros((len(env.unit_features),)),
        )
        assert jnp.allclose(
            obs["enemy_0"][0 : len(env.unit_features)],
            jnp.array([1.0, 0.00752163, 0.5390887, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
        )
        assert jnp.allclose(
            obs["enemy_0"][first_enemy_idx : first_enemy_idx + len(env.unit_features)],
            jnp.zeros((len(env.unit_features),)),
        )
        # test a dead agent sees nothing
        unit_alive = state.unit_alive.at[0].set(0)
        unit_health = state.unit_health.at[0].set(0)
        # test that the right unit corresponds to the right agent
        unit_positions = state.unit_positions.at[0].set(jnp.array([1.0, 1.0]))
        unit_positions = unit_positions.at[env.num_allies].set(jnp.array([1.0, 2.0]))
        unit_positions = unit_positions.at[env.num_allies + 1].set(
            jnp.array([1.5, 3.0])
        )
        state = state.replace(
            unit_alive=unit_alive,
            unit_health=unit_health,
            unit_positions=unit_positions,
        )
        key, key_actions = jax.random.split(key)
        actions = get_random_actions(key_actions, env)
        actions["enemy_0"] = 0
        actions["enemy_1"] = 0

        key, key_step = jax.random.split(key)

        obs, state, _, _, _ = env.step(key_step, state, actions)
        assert jnp.allclose(obs["ally_0"], jnp.zeros((env.obs_size,)))
        assert jnp.allclose(
            obs["enemy_0"][first_enemy_idx : first_enemy_idx + len(env.unit_features)],
            jnp.zeros((len(env.unit_features),)),
        )
        last_ally_idx = first_enemy_idx - len(env.unit_features)
        assert jnp.allclose(
            obs["enemy_0"][last_ally_idx : last_ally_idx + len(env.unit_features)],
            jnp.array([1.0, 0.125, 0.25, 0, 1, 0, -0.5, 1, 0, 0, 0, 0, 0]),
            atol=1e-07,
        )


@pytest.mark.parametrize("do_jit", [True, False])
def test_world_state(do_jit):
    with jax.disable_jit(do_jit):
        key = jax.random.PRNGKey(0)
        key, key_reset = jax.random.split(key)
        env, _, state = create_env(key_reset)
        start_positions = jnp.concatenate(
            [
                jnp.stack([jnp.array([8.0, 16.0])] * env.num_allies),
                jnp.stack([jnp.array([24.0, 16.0])] * env.num_enemies),
            ]
        )
        state = state.replace(unit_positions=start_positions)
        unit_1_idx = 0
        unit_2_idx = env.num_allies

        unit_alive = jnp.zeros((env.num_agents,), dtype=jnp.bool_)
        unit_alive = unit_alive.at[unit_1_idx].set(1)
        unit_alive = unit_alive.at[unit_2_idx].set(1)

        unit_health = jnp.zeros((env.num_agents,))
        unit_health = unit_health.at[unit_1_idx].set(0.5)
        unit_health = unit_health.at[unit_2_idx].set(0.5)

        state = state.replace(unit_alive=unit_alive, unit_health=unit_health)

        unit_1_action = 0
        unit_2_action = 2

        key, key_actions = jax.random.split(key)
        actions = get_random_actions(key_actions, env)
        actions[f"ally_{unit_1_idx}"] = unit_1_action
        actions[f"enemy_{unit_2_idx - env.num_allies}"] = unit_2_action

        key, key_step = jax.random.split(key)
        obs, state, _, _, _ = env.step(key_step, state, actions)

        world_state = jnp.zeros((env.state_size,))
        world_state = world_state.at[0 : len(env.own_features)].set(
            jnp.array([0.5, 8.108914, 18.952662, -0.5, 1, 0, 0, 0, 0, 0])
        )
        idx = env.num_allies * len(env.own_features)
        end_idx = idx + len(env.own_features)
        world_state = world_state.at[idx:end_idx].set(
            jnp.array([0.5, 24.217829, 12.844673, -0.5, 1, 0, 0, 0, 0, 0])
        )
        idx = env.num_agents * len(env.own_features) + env.num_allies
        end_idx = idx + env.num_enemies
        world_state = world_state.at[idx:end_idx].set(jnp.ones((env.num_enemies,)))
        assert jnp.allclose(obs["world_state"], world_state)
