import os
import subprocess
import sys


def run_script(script_path, *args):
    result = subprocess.run(
        [sys.executable, script_path, *args], capture_output=True, text=True
    )
    return result


def test_script_with_arguments():
    script_path = os.path.join("baselines/IPPO/ippo_rnn_hanabi.py")
    result = run_script(
        script_path,
        "NUM_ENVS=8",
        "NUM_STEPS=16",
        "TOTAL_TIMESTEPS=256",
        "WANDB_MODE=disabled",
    )
    assert result.returncode == 0, result.stderr
