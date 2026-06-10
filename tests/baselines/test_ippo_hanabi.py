import os
import subprocess
import sys


def run_script(script_path, *args):
    result = subprocess.run(
        [sys.executable, script_path, *args], capture_output=True, text=True
    )
    return result


def test_script_with_arguments():
    script_path = os.path.join("baselines/IPPO/ippo_ff_hanabi.py")
    result = run_script(script_path, "TOTAL_TIMESTEPS=1e4", "WANDB_MODE=disabled")
    assert result.returncode == 0, result.stderr
