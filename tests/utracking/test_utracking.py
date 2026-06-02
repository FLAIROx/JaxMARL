"""
Tests for the UTracking environment.

Covers:
    - agent movement (world_step kinematics)
    - observation exchange between agents
    - the reward function (follow / tracking / crash)
    - particle-filter prediction quality when agents are close to a moving target
    - prediction improvement when two agents track the same target
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxmarl import make

KEY = jax.random.PRNGKey(0)


def make_env(**kwargs):
    """Build a UTracking env with deterministic-friendly defaults for testing."""
    defaults = dict(
        num_agents=2,
        num_landmarks=1,
        difficulty="easy",
        pre_init_pos=False,  # avoid precomputing 100k positions (slow) for tests
        pf_num_particles=5000,
    )
    defaults.update(kwargs)
    return make("utracking", **defaults)


# --------------------------------------------------------------------------- #
# 1. Agent movement
# --------------------------------------------------------------------------- #
def test_agent_moves_as_expected():
    """world_step should move an agent by vel*dt along its (updated) heading."""
    env = make_env(num_agents=1, traj_noise_std=0.0, velocity_noise_std=0.0)
    _, state = env.reset(KEY)

    # place the agent at the origin pointing along +x (angle 0)
    pos = state.pos.at[0].set(jnp.array([0.0, 0.0, 0.0, 0.0]))
    actions = jnp.zeros(env.num_entities)  # 0.0 -> go straight

    new_pos = env.world_step(
        KEY, actions, pos, state.vel, state.traj_coeffs, state.traj_intercepts
    )

    coeff = float(env.traj_coeffs_agent[0])
    intercept = float(env.traj_intercepts_agent[0])
    vel = float(state.vel[0])

    expected_angle = ((0.0 + 0.0 * coeff + intercept) + np.pi) % (2 * np.pi) - np.pi
    expected_x = np.cos(expected_angle) * vel * env.dt
    expected_y = np.sin(expected_angle) * vel * env.dt

    assert np.isclose(float(new_pos[0, 0]), expected_x, atol=1e-3)
    assert np.isclose(float(new_pos[0, 1]), expected_y, atol=1e-3)
    # speed is fixed (no noise): distance travelled == vel * dt
    dist = float(jnp.sqrt(new_pos[0, 0] ** 2 + new_pos[0, 1] ** 2))
    assert np.isclose(dist, vel * env.dt, atol=1e-3)


def test_turn_action_changes_heading():
    """A non-zero action should rotate the agent's heading by action*coeff+intercept."""
    env = make_env(num_agents=1, traj_noise_std=0.0, velocity_noise_std=0.0)
    _, state = env.reset(KEY)

    pos = state.pos.at[0].set(jnp.array([0.0, 0.0, 0.0, 0.0]))
    action = 0.24
    actions = jnp.array([action, 0.0])

    new_pos = env.world_step(
        KEY, actions, pos, state.vel, state.traj_coeffs, state.traj_intercepts
    )

    coeff = float(env.traj_coeffs_agent[0])
    intercept = float(env.traj_intercepts_agent[0])
    expected_angle = ((0.0 + action * coeff + intercept) + np.pi) % (2 * np.pi) - np.pi
    assert np.isclose(float(new_pos[0, 3]), expected_angle, atol=1e-4)


# --------------------------------------------------------------------------- #
# 2. Observation exchange
# --------------------------------------------------------------------------- #
def test_observations_exchanged_correctly():
    """An agent should observe the true relative position of the other agent
    (no noise, no dropped communication)."""
    env = make_env(
        num_agents=2,
        num_landmarks=1,
        agent_obs_noise_std=0.0,
        lost_comm_prob=0.0,
        max_comm_dist=1e9,
        steps_for_agent_communication=1,
        matrix_obs=True,
        space_unit=1.0,  # disable rescaling to read raw deltas
    )
    obs, state = env.reset(KEY)

    o1 = obs["agent_1"]  # shape (num_entities, obs_feats)
    # entity 0 is agent_1 itself, entity 1 is agent_2, entity 2 is the landmark
    assert o1[0, 5] == 1.0  # is_self for agent_1
    assert o1[1, 5] == 0.0
    assert o1[0, 4] == 1.0  # agents have is_agent == 1
    assert o1[1, 4] == 1.0
    assert o1[2, 4] == 0.0  # landmark has is_agent == 0

    # relative position of agent_2 as seen by agent_1 == pos[agent_1] - pos[agent_2]
    expected = np.array(state.pos[0, :3] - state.pos[1, :3])
    assert np.allclose(np.array(o1[1, :3]), expected, atol=1e-3)


def test_dropped_communication_keeps_stale_position():
    """With communication always lost, the other agent's observed position must
    not update after a step (stale value retained)."""
    env = make_env(
        num_agents=2,
        num_landmarks=1,
        agent_obs_noise_std=0.0,
        lost_comm_prob=1.0,  # always drop
        steps_for_agent_communication=1,
        matrix_obs=True,
        space_unit=1.0,
    )
    obs0, state = env.reset(KEY)
    before = np.array(obs0["agent_1"][1, :3])

    actions = {a: jnp.array(2) for a in env.agents}  # straight
    obs1, _, _, _, _ = env.step_env(KEY, state, actions)
    after = np.array(obs1["agent_1"][1, :3])

    assert np.allclose(before, after, atol=1e-5)


# --------------------------------------------------------------------------- #
# 3. Reward function
# --------------------------------------------------------------------------- #
def test_reward_follow_within_threshold():
    """A landmark within rew_dist_thr of an agent yields the full follow reward."""
    env = make_env(
        num_agents=1,
        num_landmarks=1,
        rew_follow_coeff=1.0,
        rew_tracking_coeff=0.0,
        rew_norm_landmarks=True,
        rew_dist_thr=180.0,
    )
    pos = jnp.array([[0.0, 0.0, 0.0, 0.0], [50.0, 0.0, 5.0, 0.0]])
    ranges = jnp.array([[0.0, 50.0]])
    ranges_real_2d = jnp.array([[0.0, 50.0]])
    land_pred_pos = jnp.array([[[50.0, 0.0, 5.0]]])  # perfect prediction

    rew, _, info = env.get_rew_done_info(0, pos, ranges, ranges_real_2d, land_pred_pos, 0.0)
    assert np.isclose(float(rew), 1.0, atol=1e-5)
    assert float(info["follow_rew"]) == 1.0


def test_reward_no_follow_when_far():
    """A landmark beyond rew_dist_thr gives zero follow reward."""
    env = make_env(
        num_agents=1,
        num_landmarks=1,
        rew_follow_coeff=1.0,
        rew_tracking_coeff=0.0,
        rew_dist_thr=180.0,
    )
    pos = jnp.array([[0.0, 0.0, 0.0, 0.0], [300.0, 0.0, 5.0, 0.0]])
    ranges = jnp.array([[0.0, 300.0]])
    ranges_real_2d = jnp.array([[0.0, 300.0]])
    land_pred_pos = jnp.array([[[300.0, 0.0, 5.0]]])

    rew, _, _ = env.get_rew_done_info(0, pos, ranges, ranges_real_2d, land_pred_pos, 0.0)
    assert np.isclose(float(rew), 0.0, atol=1e-5)


def test_reward_tracking_decay():
    """Tracking reward is 1 for a perfect prediction and 0 above the threshold."""
    env = make_env(
        num_agents=1,
        num_landmarks=1,
        rew_follow_coeff=0.0,
        rew_tracking_coeff=1.0,
        rew_norm_landmarks=True,
        rew_pred_thr=50.0,
    )
    pos = jnp.array([[0.0, 0.0, 0.0, 0.0], [50.0, 0.0, 5.0, 0.0]])
    ranges = jnp.array([[0.0, 50.0]])
    ranges_real_2d = jnp.array([[0.0, 50.0]])

    perfect = jnp.array([[[50.0, 0.0, 5.0]]])
    rew_perfect, _, _ = env.get_rew_done_info(0, pos, ranges, ranges_real_2d, perfect, 0.0)
    assert np.isclose(float(rew_perfect), 1.0, atol=1e-5)

    bad = jnp.array([[[200.0, 0.0, 5.0]]])  # 150 m error >> threshold
    rew_bad, _, _ = env.get_rew_done_info(0, pos, ranges, ranges_real_2d, bad, 0.0)
    assert np.isclose(float(rew_bad), 0.0, atol=1e-5)


def test_reward_crash_penalty():
    """Two agents closer than min_valid_distance trigger the crash penalty (-1)."""
    env = make_env(
        num_agents=2,
        num_landmarks=1,
        penalty_for_crashing=True,
        min_valid_distance=20.0,
    )
    # agents 10 m apart, landmark far away
    pos = jnp.array(
        [[0.0, 0.0, 0.0, 0.0], [10.0, 0.0, 0.0, 0.0], [500.0, 0.0, 5.0, 0.0]]
    )
    ranges = jnp.array([[0.0, 10.0, 500.0], [10.0, 0.0, 500.0]])
    ranges_real_2d = ranges
    land_pred_pos = jnp.tile(jnp.array([500.0, 0.0, 5.0]), (2, 1, 1))

    rew, _, info = env.get_rew_done_info(0, pos, ranges, ranges_real_2d, land_pred_pos, 0.0)
    assert np.isclose(float(rew), -1.0, atol=1e-5)
    assert bool(info["unsafe_distance"][0])


# --------------------------------------------------------------------------- #
# 4 & 5. Particle-filter prediction quality
# --------------------------------------------------------------------------- #
def _pinned_rollout(env, key, n_steps, offsets):
    """Roll out the env keeping each agent pinned at a fixed offset from the
    (moving) landmark, and return the per-step tracking error."""
    obs, state = env.reset(key)
    errors = []
    actions = {a: jnp.array(2) for a in env.agents}  # straight
    for _ in range(n_steps):
        key, k = jax.random.split(key)
        land_xy = state.pos[env.num_agents :, :2]
        pos = state.pos
        for ai in range(env.num_agents):
            lx, ly = land_xy[0]  # single landmark
            ox, oy = offsets[ai]
            pos = pos.at[ai, 0].set(lx + ox)
            pos = pos.at[ai, 1].set(ly + oy)
        state = state.replace(pos=pos)
        obs, state, _, _, info = env.step_env(k, state, actions)
        errors.append(float(info["tracking_error_mean"][0]))
    return np.array(errors)


def _mean_late_error(env_kwargs, offsets, n_steps=50, late=25, seeds=4):
    """Average tracking error over the last `late` steps across several seeds."""
    env = make_env(**env_kwargs)
    vals = []
    for s in range(seeds):
        errs = _pinned_rollout(env, jax.random.PRNGKey(s), n_steps, offsets)
        vals.append(errs[-late:].mean())
    return float(np.mean(vals))


def test_prediction_good_when_close_to_moving_target():
    """A single agent staying close to a moving target predicts it well, and
    much better than an agent that is too far to get usable ranges."""
    close = _mean_late_error(dict(num_agents=1, num_landmarks=1), offsets=[(30.0, 0.0)])
    far = _mean_late_error(dict(num_agents=1, num_landmarks=1), offsets=[(1000.0, 0.0)])

    # close prediction should be reliable (well below the initial spawn distance)
    assert close < 200.0, f"close prediction error too high: {close:.1f} m"
    # and clearly better than when the target is out of usable range
    assert close < far, f"close ({close:.1f}) not better than far ({far:.1f})"


def test_prediction_better_with_two_agents():
    """Two agents tracking the same moving target from different angles predict
    its position better than a single agent."""
    one = _mean_late_error(dict(num_agents=1, num_landmarks=1), offsets=[(30.0, 0.0)])
    two = _mean_late_error(
        dict(num_agents=2, num_landmarks=1), offsets=[(30.0, 0.0), (0.0, 30.0)]
    )
    assert two < one, f"two-agent error ({two:.1f}) not better than one ({one:.1f})"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
