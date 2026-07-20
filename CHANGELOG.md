NOTE: We use effort-based-versioning

### [v0.2.0]

**Several breaking changes**. JAX/CUDA compatibility update, jaxtyping type annotations, State variable name changes for consistency across environments, quality of life improvements, and bug fixes.

#### Breaking Changes
**Hanabi**:
 - `HanabiState` now inherits from `BaseState`; `turn` field renamed to `step` and `done` field added (`terminal | out_of_lives`). Code accessing `state.turn` must be updated to `state.step`
**SMAX**:
 - Dead units no longer move or participate in collision resolution, `update_position` freezes dead unit positions; `_push_units_away` gates the overlap term by an alive-alive pair mask and preserves dead unit positions;
 - stale hardcoded test values updated to reflect both fixes and JAX PRNG changes (`test_obs_function` was subsequently made PRNG-independent, see Fixed).
 - `last_targeted` and `weapon_cooldown` observation features were un-normalised, violating the declared `Box(low=-1, high=1)` space. `last_targeted` is now divided by `(max_actions - 1)` and `weapon_cooldown` is normalised by per-unit-type max cooldown and clipped to `[0, 1]` (**existing trained agents will observe different values for these features**)
 - `State` now inherits from `BaseState`; `time` field renamed to `step`, `terminal` field renamed to `done`. Code accessing `state.time` or `state.terminal` must be updated
**Overcooked**:
 - Deprecation noticed added to env __init__ as overcookedv2 should be preferred.
 - `State` now inherits from `BaseState`; `time` field renamed to `step`, `terminal` field renamed to `done`. Code accessing `state.time` or `state.terminal` must be updated
**OvercookedV2**:
 - `State` now inherits from `BaseState`; `time` field renamed to `step`, `terminal` field renamed to `done`; class decorator changed from `@chex.dataclass` to `@struct.dataclass`. Code accessing `state.time` or `state.terminal` must be updated
**Storm** (`InTheGrid`, `InTheGrid_2p`, `InTheMatrix`):
 - `State` now inherits from `BaseState`; `inner_t` field renamed to `step`, `done` field added (true when the outer episode ends). Code accessing `state.inner_t` must be updated to `state.step`
**CoinGame**:
 - `EnvState` now inherits from `BaseState`; class decorator changed from `@chex.dataclass` to `@struct.dataclass`; `inner_t` field renamed to `step`, `done` field added. Code accessing `state.inner_t` must be updated to `state.step`
 - **`dones` (and `state.done`) now mark the outer/meta episode boundary, not the inner one**, matching Storm and the rest of the base API. Previously `dones["__all__"]` fired every `num_inner_steps`; it now fires once per meta episode, after `num_outer_steps` inner episodes. The inner boundary is reported as `info["inner_episode_done"]` — algorithms that truncate GAE or reset recurrent state at inner resets must read that key instead of `dones`. `LogWrapper` consequently reports the full iterated-game return rather than per-inner-episode returns
 - The `self.step = _step` bypass of the base class is removed: the transition is now `step_env` and the outer boundary is handled by the standard `MultiAgentEnv.step` auto-reset. `get_obs` is now implemented, so `step(..., reset_state=...)` works
**MABrax**:
 - Deprecated due to Brax itself being deprecated; this environment needs migrating to MJX.
 - Type annotations added.
 - brax install moved to optional extras: `pip install jaxmarl[mabrax]` due to deprecation; `jaxmarl.environments.mabrax` will raise an ImportError if brax is not installed.
**Wrappers**:
 - `_batchify_floats` renamed to `stack_agent_values`

**Repo wide**:
JAX: `jax<=0.4.38` upper cap removed, floor set to `>=0.4.25` (`jax.tree` became a public API in 0.4.25); `jaxlib` removed (transitive dep of jax)
- `brax==0.10.3` → `brax>=0.10.3`, `flashbax==0.1.0` → `flashbax>=0.1.0` (relaxed exact pins)
- `mujoco` and `scipy` removed from explicit deps (transitive)
- `numpy>=1.26` added as explicit dep
- `pettingzoo` removed from `algs`; `mpe2` added to `dev`; MPE test files migrated
- Python minimum raised to 3.11 (`jaxtyping>=0.2.28` dropped active support below 3.11; CI matrix targets 3.11/3.12 matching the dev container)
- Dockerfile: base image updated to `nvcr.io/nvidia/jax:26.04-py3`; apt cache cleared; `XLA_PYTHON_CLIENT_MEM_FRACTION` removed (unnecessary alongside `PREALLOCATE=false`)

#### Added
- **Type annotations on `MultiAgentEnv` and all environment classes**; `jaxtyping>=0.2.28` added as core dependency
- `MultiAgentEnv` and `SUBMODULE_ENVIRONMENTS` now importable directly from `jaxmarl.registration`
- `jaxmarl/py.typed`: package now ships type information for IDE support
- `[tool.pyright]` config in `pyproject.toml` scoped to public API files
- `.pre-commit-config.yaml`: ruff linting and formatting enforced on commit; switched from `astral-sh/ruff-pre-commit` to `language: system` to use the project-installed ruff and avoid network fetches on first commit
- `.devcontainer/devcontainer.json`: one-click VS Code dev container with GPU support
- `.github/workflows/ci.yml`: lightweight CI (lint, typecheck, pytest on Python 3.11/3.12) without Docker
- `.github/workflows/docs.yml`: auto-deploy MkDocs docs to GitHub Pages on push to `main`
- GitHub issue templates (bug report, feature request) and PR checklist
- `make local-test` target for running tests without Docker

#### Fixed
- `docs/index.md` environment count: 9 → 11
- `Dict.contains` in `spaces.py`: `getattr(x, k)` → `x[k]`
- `tests/brax/test_brax_rand_acts.py`: skipped due to native double-free crash in legacy brax environments (unmaintained, incompatible with current jaxlib)
- **MPE tests**: `env_zoo.reset()` added at the start of each episode in `tests/mpe/test_mpe.py` to fix `AttributeError: 'aec_to_parallel_wrapper' object has no attribute 'agents'` with current PettingZoo
- README installation instructions improved.
- **Hanabi** type annotations updated to the JAX conventions: `chex.Array`/`chex.PRNGKey` replaced by `jax.Array`/`jaxtyping.PRNGKeyArray`, `reset`/`step_env`/`get_obs` typed with the `Observations`/`Actions`/`Rewards`/`Dones`/`Infos` aliases, and the `State` counter fields (`num_cards_dealt`, `num_cards_discarded`, `last_round_count`, `score`) typed `jax.Array` to match the traced values assigned to them.
- **Overcooked / OvercookedV2** type annotations updated to the JAX conventions: `chex.Array` and `jnp.ndarray` replaced by `jax.Array` throughout (`State` fields, `step_agents`, `process_interact`, `common.py` `Position`/`Agent`, `utils.py`), `is_terminal` now returns `jax.Array` rather than `bool`, and scalar `State` fields (`step`, `done`, `recipe`, `new_correct_delivery`) are constructed as arrays. `OvercookedV2.state_space()` raised `TypeError` when `agent_view_size` was `None` (the default) - it now treats an unset view size as `0`. `OvercookedVisualizer`/`OvercookedV2Visualizer` `window` attribute typed `Optional[Window]`, `_lazy_init_window` returns the window, and `close()` no longer fails when no window was opened.
- **SMAX** type annotations updated to the JAX conventions: `chex.Array` replaced by `jax.Array` throughout `smax_env.py` and `heuristic_enemy.py` (both `import chex` dropped), unvectorised index parameters of `_observe_features`/`_get_own_features`/the `get_features` closures retyped from `int` to `jax.Array` (they are traced under `jax.vmap`), `HeuristicPolicyState.last_attacked_enemy` typed `Union[int, jax.Array]` to match the `jax.lax.select` result assigned to it, and `PRNGKeyArray` applied to the remaining untyped `key` parameters in `distributions.py` and `heuristic_enemy_smax_env.py`.
- **SMAX** `tests/smax/test_smax.py::test_obs_function` no longer depends on the positions drawn at reset. The expected observations were recorded under a JAX build where `jax_threefry_partitionable` defaults to `True`, so the test failed on any JAX where it defaults to `False` (e.g. 0.4.37, permitted by the `jax>=0.4.25` floor). The test now pins `unit_positions` before asserting, as `test_world_state` already did, and passes under both PRNG implementations.
- **MPE** type annotations updated to the JAX conventions across `simple.py` and every `simple_*.py`: `chex.Array` replaced by `jax.Array` (`import chex` dropped), `State.done`/`State.step` constructed as `jnp.array(...)` rather than Python `False`/`0`, and the index parameters of vmapped helpers (`_observation`, `_reward`, `_decode_continuous_action`, `_decode_discrete_action`, `_collisions`, `_common_stats`) retyped from `int` to `jax.Array` since they are traced under `jax.vmap`. Optional state fields (`goal`, and `goal_colour`/`private_key` in `simple_crypto.py`) are now `Optional[jax.Array]`, extracted to locals with a trace-time `assert ... is not None` at the sites that use them arithmetically. `SimpleMPE.set_actions` no longer rebinds its `actions` parameter to an array (a parameter's declared type cannot be redeclared in its own body); the stacked array is a separate local.
- **CoinGame** type annotations updated to the JAX conventions: `EnvState.outer_t` retyped from `int`, `payoff_matrix` annotated `List[List[float]]`, `_abs_position`/`_state_to_obs` declared as returning `Observations`, and `state2idx` as returning an array rather than `int`. `render` now converts positions to Python floats through an `_xy` helper before passing them to matplotlib. Two hardcoded values in the egocentric observation path were also corrected: the colour-swapped `EnvState` built by `_state_to_obs` now carries `step`/`outer_t`/`done` from the source state instead of `0`/`0`/`False`, and `step_env` propagates `state.done` instead of hardcoding `False`. The `shared_rewards` branch is simplified to `red_reward + blue_reward` (equivalent to the previous `sum`-over-`zip` construction).
- **Gridworld** `GOAL_COLOR_TO_INDEX` constant added to `common.py`; `goal_color_sequence=None` parameter added to `make_maze_map` and `make_overcooked_map`; `sample_n_walls=False` added to `Maze` and `MAMaze` `__init__`; bare `window` reference in `interactive.py` fixed to `extras["viz"].window`.

### [v0.1.0]

Several environment (Hanabi, Storm, Overcooked, Coin Game, MPE FacMac) API's updated to conform with the base API, new test added to ensure this happens for future environments. As this will change user interactions with some environments, have updated version to 0.1.0 to reflect the possible effort needed. Baseline scripts also updated to reflect this change.

#### Added
 - Overcooked v2

### [v0.0.4]

#### Added
 - Reset to specific state within environment base class
 - JaxNav environment


### [v0.0.3]

#### Added
 - Hanabi bug fixes

### [v0.0.1]

##### Added
 - base set of environments and algorithms
