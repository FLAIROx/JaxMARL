#!/usr/bin/env python3
"""
Runs a set of JaxMARL baselines distributed across GPU slots and logs to WandB, where
seed gets its own WandB run. Seeds for each (algo, env) run concurrently across GPUs;
combos themselves run sequentially.

Settings (GPUs, seeds, step counts, WandB project) are configured in `baselines/config/run_minimal_baseline_set.yaml`
and can be overridden on the command line via Hydra.


# Single GPU, all baselines
`python baselines/run_minimal_baseline_set.py entity=<wandb-entity>`

# Two GPUs, two concurrent seeds per GPU for lightweight envs
`python baselines/run_minimal_baseline_set.py entity=<wandb-entity> 'gpus=[0,1]' seeds_per_gpu.ippo_ff_mpe=2`

# Quick smoke test: one baseline, WandB disabled, minimal steps
`python baselines/run_minimal_baseline_set.py entity=<wandb-entity> 'only=[ippo_ff_mpe]' wandb_mode=disabled steps=1000`


Can also run in the Docker Container via Make:
`make run-baseline-set ARGS="entity=<wandb-entity> 'gpus=[0,1]'"`


"""

import os
import queue
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_LOG_DIR = REPO_ROOT / ".baseline_logs" / "runs"


def jaxmarl_version() -> str:
    """Read jaxmarl's __version__ WITHOUT importing it.

    Importing jaxmarl pulls in JAX, which initialises CUDA and preallocates its
    default 75% of a GPU in this (launcher) process -- ~18.4GB of a 24GB card.
    That memory is then unavailable to the training subprocesses scheduled onto
    that GPU, which die with RESOURCE_EXHAUSTED.

    Hiding the GPUs from the parent instead does not work: jaxmarl evaluates
    jnp.array() at import time and raises with no backend available. Setting
    JAX_PLATFORMS=cpu would be inherited by the children and force them onto
    CPU. So the only safe option is to never import jaxmarl here.
    """
    src = (REPO_ROOT / "jaxmarl" / "__init__.py").read_text()
    m = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', src, re.M)
    if not m:
        raise RuntimeError("could not parse __version__ from jaxmarl/__init__.py")
    return m.group(1)


JAXMARL_VERSION = jaxmarl_version()

# Maps (algo, suite) -> (script_relpath, alg_key, alg_prefix).
#
# alg_key: name of the Hydra config group file under <script_dir>/config/alg/.
#          Empty string for flat configs (IPPO/MAPPO) that bake env settings into
#          the script's own YAML.
# alg_prefix: Hydra key prefix for CLI overrides. "alg." for QLearning scripts
#             that use a nested alg config group; "" for flat configs.
ALGO_REGISTRY: dict[tuple[str, str], tuple[str, str, str]] = {
    # MPE
    ("ippo_ff", "mpe"): ("baselines/IPPO/ippo_ff_mpe.py", "", ""),
    ("ippo_rnn", "mpe"): ("baselines/IPPO/ippo_rnn_mpe.py", "", ""),
    ("mappo_rnn", "mpe"): ("baselines/MAPPO/mappo_rnn_mpe.py", "", ""),
    ("qmix_rnn", "mpe"): ("baselines/QLearning/qmix_rnn.py", "ql_rnn_mpe", "alg."),
    ("vdn_rnn", "mpe"): ("baselines/QLearning/vdn_rnn.py", "ql_rnn_mpe", "alg."),
    ("vdn_ff", "mpe"): ("baselines/QLearning/vdn_ff.py", "vdn_ff_mpe", "alg."),
    # SMAX — flat configs use MAP_NAME; QLearning nested configs use alg.MAP_NAME
    ("ippo_rnn", "smax"): ("baselines/IPPO/ippo_rnn_smax.py", "", ""),
    ("mappo_rnn", "smax"): ("baselines/MAPPO/mappo_rnn_smax.py", "", ""),
    ("qmix_rnn", "smax"): ("baselines/QLearning/qmix_rnn.py", "ql_rnn_smax", "alg."),
    ("vdn_rnn", "smax"): ("baselines/QLearning/vdn_rnn.py", "ql_rnn_smax", "alg."),
    # Overcooked — env field is "overcooked_{layout}"; we pass ENV_KWARGS.layout
    ("ippo_ff", "overcooked"): ("baselines/IPPO/ippo_ff_overcooked.py", "", ""),
    ("vdn_cnn", "overcooked"): (
        "baselines/QLearning/vdn_cnn_overcooked.py",
        "ql_cnn_overcooked",
        "alg.",
    ),
    # Hanabi — ENV_NAME is always "hanabi"; the env field is a descriptive label only
    ("ippo_ff", "hanabi"): ("baselines/IPPO/ippo_ff_hanabi.py", "", ""),
    ("mappo_ff", "hanabi"): ("baselines/MAPPO/mappo_ff_hanabi.py", "", ""),
}


def suite_env_args(suite: str, env: str, alg_prefix: str) -> list[str]:
    """CLI overrides that select the environment within a suite."""
    p = alg_prefix
    if suite == "mpe":
        return [f"{p}ENV_NAME={env}"]
    if suite == "smax":
        # SMAX scripts use MAP_NAME for the scenario; ENV_NAME is fixed in config.
        return [f"{p}MAP_NAME={env}"]
    if suite == "overcooked":
        # env is e.g. "overcooked_coord_ring"; layout is the suffix after "overcooked_".
        layout = env.removeprefix("overcooked_")
        return [f"{p}ENV_KWARGS.layout={layout}"]
    if suite == "hanabi":
        # Scripts have ENV_NAME=hanabi baked in; env field is just a label.
        return []
    return [f"{p}ENV_NAME={env}"]


def expand_runs(suites_cfg: dict) -> list[dict]:
    """Expand the suites config into a flat list of (suite, env, algo) dicts."""
    runs = []
    for suite_name, suite in suites_cfg.items():
        seeds_per_gpu = suite["seeds_per_gpu"]
        for entry in suite["envs"]:
            env = entry["env"]
            for algo in entry["algos"]:
                reg = ALGO_REGISTRY.get((algo, suite_name))
                run: dict = {
                    "name": f"{algo}_{env}",
                    "suite": suite_name,
                    "env": env,
                    "algo": algo,
                    "seeds_per_gpu": seeds_per_gpu,
                }
                if reg is None:
                    run["implemented"] = False
                else:
                    script, alg_key, alg_prefix = reg
                    run.update(
                        implemented=True,
                        script=script,
                        alg_key=alg_key,
                        alg_prefix=alg_prefix,
                    )
                runs.append(run)
    return runs


def validate_runs(runs: list[dict]) -> bool:
    """Print a table of all runs and verify scripts and alg configs exist on disk."""
    col_w = max(len(r["name"]) for r in runs) + 2
    ok = True

    print(f"\n{'─' * 74}")
    print(f"  {'Run':<{col_w}}  {'Status':<10}  Details")
    print(f"{'─' * 74}")

    for run in runs:
        name = run["name"]
        if not run["implemented"]:
            print(
                f"  {name:<{col_w}}  NOT IMPL   no registry entry for ({run['algo']}, {run['suite']})"
            )
            ok = False
            continue

        errors: list[str] = []
        script_path = REPO_ROOT / run["script"]
        if not script_path.exists():
            errors.append(f"script missing: {run['script']}")

        alg_key = run["alg_key"]
        if alg_key:
            alg_cfg = script_path.parent / "config" / "alg" / f"{alg_key}.yaml"
            if not alg_cfg.exists():
                errors.append(f"alg config missing: {alg_cfg.relative_to(REPO_ROOT)}")

        if errors:
            for e in errors:
                print(f"  {name:<{col_w}}  ERROR      {e}")
            ok = False
        else:
            detail = run["script"] + (f"  +alg={alg_key}" if alg_key else "")
            print(f"  {name:<{col_w}}  OK         {detail}")

    print(f"{'─' * 74}\n")
    return ok


_print_lock = threading.Lock()


def tprint(*args, **kwargs) -> None:
    with _print_lock:
        print(*args, **kwargs)


def build_cmd(
    run: dict,
    *,
    project: str,
    entity: str,
    wandb_mode: str,
    seed: int,
) -> list[str]:
    p = run["alg_prefix"]
    cmd = [sys.executable, str(REPO_ROOT / run["script"])]
    if run["alg_key"]:
        cmd.append(f"+alg={run['alg_key']}")
    cmd += [
        f"PROJECT={project}",
        f"ENTITY={entity}",
        f"WANDB_MODE={wandb_mode}",
        f"SEED={seed}",
        "NUM_SEEDS=1",
    ]
    cmd += suite_env_args(run["suite"], run["env"], p)
    return cmd


def run_combo(
    run: dict,
    *,
    project: str,
    entity: str,
    wandb_mode: str,
    seed: int,
    seeds: int,
    gpus: list[int],
    seeds_per_gpu: int,
    tag_jaxmarl_version: bool,
) -> tuple[str, bool, float]:
    algo_family, network = run["algo"].rsplit("_", 1)
    tags = [algo_family.upper(), network.upper(), run["env"], run["suite"].upper()]
    if tag_jaxmarl_version:
        tags.append(f"jaxmarl-{JAXMARL_VERSION}")
    wandb_tags = ",".join(tags)
    # Use dashes in WandB names for readability; underscore in run["name"] for key use.
    wandb_group = f"{run['algo']}-{run['env']}"
    if tag_jaxmarl_version:
        wandb_group += f"-jaxmarl_{JAXMARL_VERSION}"

    gpu_pool: queue.Queue[int] = queue.Queue()
    for gpu in gpus:
        for _ in range(seeds_per_gpu):
            gpu_pool.put(gpu)

    t0 = time.monotonic()
    passed = True
    RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)

    def run_seed(i: int) -> bool:
        gpu_id = gpu_pool.get()
        try:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            # Always disable preallocation, not just when sharing a GPU. With it
            # on, a lone run still grabs 75% of the card (~18.4GB of 24GB) which
            # barely covers a full-scale SMAX run and OOMs on any spike.
            env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
            env["WANDB_TAGS"] = wandb_tags
            env["WANDB_RUN_GROUP"] = wandb_group
            env["WANDB_NAME"] = f"{wandb_group}-{i + 1}"
            cmd = build_cmd(
                run,
                project=project,
                entity=entity,
                wandb_mode=wandb_mode,
                seed=seed + i,
            )
            tprint(f"  [{run['name']}] seed {i + 1}/{seeds} starting on GPU {gpu_id}")
            # Per-run log: interleaved stdout is unreadable across concurrent
            # seeds, and a failure with no retained output is undiagnosable.
            log_path = RUN_LOG_DIR / f"{wandb_group}-{i + 1}.log"
            with open(log_path, "w") as fh:
                result = subprocess.run(
                    cmd, cwd=REPO_ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT
                )
            if result.returncode != 0:
                tprint(
                    f"  [{run['name']}] seed {i + 1} FAIL "
                    f"(exit {result.returncode}) -> {log_path}"
                )
                for line in log_path.read_text().strip().splitlines()[-15:]:
                    tprint(f"      | {line}")
                return False
            return True
        except Exception as exc:
            tprint(f"  [{run['name']}] seed {i + 1} ERROR: {exc}")
            return False
        finally:
            gpu_pool.put(gpu_id)

    total_slots = len(gpus) * seeds_per_gpu
    with ThreadPoolExecutor(max_workers=total_slots) as executor:
        futures = [executor.submit(run_seed, i) for i in range(seeds)]
        for future in as_completed(futures):
            if not future.result():
                passed = False

    elapsed = time.monotonic() - t0
    tprint(f"  [{run['name']}] {'PASS' if passed else 'FAIL'} — {elapsed:.1f}s")
    return run["name"], passed, elapsed


@hydra.main(version_base=None, config_path=".", config_name="run_minimal_baseline_set")
def main(config: DictConfig) -> None:
    suites_cfg: dict = OmegaConf.to_container(config.suites, resolve=True)  # type: ignore[assignment]
    gpus: list[int] = list(config.gpus)
    only = set(config.only) if config.only else None
    only_suites = set(config.only_suites) if config.only_suites else None
    only_algos = set(config.only_algos) if config.only_algos else None

    runs = expand_runs(suites_cfg)
    if only:
        runs = [r for r in runs if r["name"] in only]
    if only_suites:
        runs = [r for r in runs if r["suite"] in only_suites]
    if only_algos:
        runs = [r for r in runs if r["algo"] in only_algos]

    if config.dry_run:
        sys.exit(0 if validate_runs(runs) else 1)

    unimplemented = [r for r in runs if not r["implemented"]]
    if unimplemented:
        lines = "\n".join(
            f"  {r['name']}  (no registry entry for ({r['algo']}, {r['suite']}))"
            for r in unimplemented
        )
        raise ValueError(
            f"Unimplemented combos — add to ALGO_REGISTRY or remove from YAML:\n{lines}"
        )

    print(f"\nRunning {len(runs)} combo(s) → wandb project '{config.project}'")
    print(
        f"  seeds={config.seeds}  seed={config.seed}  gpus={gpus}  wandb={config.wandb_mode}\n"
    )

    run_kwargs = dict(
        project=config.project,
        entity=config.entity,
        wandb_mode=config.wandb_mode,
        seed=config.seed,
        seeds=config.seeds,
        gpus=gpus,
        tag_jaxmarl_version=config.tag_jaxmarl_version,
    )

    results: list[tuple[str, bool, float]] = []
    for r in runs:
        results.append(
            run_combo(
                r,
                seeds_per_gpu=r["seeds_per_gpu"],
                **run_kwargs,
            )
        )

    print(f"\n{'═' * 60}")
    print("  Summary")
    print(f"{'─' * 60}")
    any_failed = False
    for name, passed, elapsed in results:
        print(f"  {'PASS' if passed else 'FAIL':<6}  {elapsed:>8.1f}s  {name}")
        if not passed:
            any_failed = True
    print(f"{'═' * 60}\n")

    sys.exit(1 if any_failed else 0)


if __name__ == "__main__":
    main()
