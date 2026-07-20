"""
CoinGame is an iterated game: `num_outer_steps` inner episodes make up one meta
episode. `done` refers to the outer/meta boundary, matching Storm and the rest of
the base API; the inner boundary is reported in `info["inner_episode_done"]`.
"""

import jax
import jax.numpy as jnp
import pytest

from jaxmarl.environments.coin_game import CoinGame
from jaxmarl.wrappers.baselines import LogWrapper

NUM_INNER = 3
NUM_OUTER = 4


def _rollout(env, num_steps, seed=0):
    """Step the env with a fixed no-op action, collecting per-step outputs."""
    rng = jax.random.PRNGKey(seed)
    rng, rng_reset = jax.random.split(rng)
    _, state = env.reset(rng_reset)

    step = jax.jit(env.step)
    trajectory = []
    for _ in range(num_steps):
        rng, rng_step = jax.random.split(rng)
        actions = {a: jnp.int32(4) for a in env.agents}  # 4 == stay
        _, state, reward, done, info = step(rng_step, state, actions)
        trajectory.append((state, reward, done, info))
    return trajectory


@pytest.fixture
def env():
    return CoinGame(num_inner_steps=NUM_INNER, num_outer_steps=NUM_OUTER)


def test_done_marks_only_the_outer_boundary(env):
    trajectory = _rollout(env, NUM_INNER * NUM_OUTER)
    dones = [bool(done["__all__"]) for _, _, done, _ in trajectory]

    # exactly one meta episode in num_inner * num_outer steps, ending on the last
    assert [i + 1 for i, d in enumerate(dones) if d] == [NUM_INNER * NUM_OUTER]

    # per-agent dones agree with __all__
    for _, _, done, _ in trajectory:
        for agent in env.agents:
            assert bool(done[agent]) == bool(done["__all__"])


def test_inner_boundary_reported_in_info(env):
    trajectory = _rollout(env, NUM_INNER * NUM_OUTER)
    inner_dones = [bool(info["inner_episode_done"]) for _, _, _, info in trajectory]

    # one inner reset every num_inner steps
    assert [i + 1 for i, d in enumerate(inner_dones) if d] == [
        NUM_INNER * (k + 1) for k in range(NUM_OUTER)
    ]


def test_inner_reset_does_not_wipe_the_outer_clock(env):
    """The inner reset is internal to step_env: it advances outer_t rather than
    triggering the base class auto-reset."""
    trajectory = _rollout(env, NUM_INNER * NUM_OUTER)

    outer_ts = [int(state.outer_t) for state, _, _, _ in trajectory]
    # outer_t climbs across inner episodes, then the auto-reset returns it to 0
    assert outer_ts[NUM_INNER - 1] == 1
    assert outer_ts[2 * NUM_INNER - 1] == 2
    assert outer_ts[-1] == 0, "outer episode end should auto-reset outer_t"

    # step counts cycle within each inner episode
    steps = [int(state.step) for state, _, _, _ in trajectory]
    assert steps[: NUM_INNER + 1] == [1, 2, 0, 1]


def test_log_wrapper_logs_the_full_meta_episode(env):
    """LogWrapper cuts on __all__, so it should report the meta episode length."""
    wrapped = LogWrapper(env)
    trajectory = _rollout(wrapped, NUM_INNER * NUM_OUTER)
    _, _, _, info = trajectory[-1]

    lengths = info["returned_episode_lengths"]
    assert jnp.all(lengths == NUM_INNER * NUM_OUTER), (
        f"expected meta-episode length {NUM_INNER * NUM_OUTER}, got {lengths}"
    )
