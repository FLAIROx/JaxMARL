import subprocess
import sys
import os

def run_script(script_path, *args):
    result = subprocess.run([sys.executable, script_path, *args], capture_output=True, text=True)
    return result

def test_script_with_arguments():
    script_path = os.path.join('baselines/IPPO/ippo_ff_mabrax.py')
    result = run_script(script_path, 'TOTAL_TIMESTEPS=1e4', 'WANDB_MODE=disabled')
    print('result ', result.returncode)
    assert result.returncode == 0
    # assert "Arguments received: ['arg1', 'arg2', 'arg3']" in result.stdout


test_script_with_arguments()