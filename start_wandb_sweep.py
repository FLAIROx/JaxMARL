import jax
import yaml
import wandb
from os import system


def tmux(command):
    system('tmux %s' % command)

def tmux_shell(command):
    tmux('send-keys "%s" "C-m"' % command)

sweep_config = yaml.load(open("sweep_smax.yaml", "r"), Loader=yaml.FullLoader)
print('sweep_config:', sweep_config)

project = sweep_config["project"]

# Initialize WandB Sweep
sweep_id = wandb.sweep(sweep_config, project=project)


agents_per_gpu = 2
print('jax.devices():', jax.devices())
# count number of GPUs

gpu_count = [1 for device in jax.devices() if 'gpu' in device.platform.lower()]
gpu_count = sum(gpu_count)

gpus_to_use = "all"
if gpus_to_use != "all":
    gpu_count = min(gpu_count, gpus_to_use)
    if gpu_count < gpus_to_use:
        print(f"Warning: only {gpu_count} GPUs available, but {gpus_to_use} requested.")

print('gpu_count:', gpu_count)
# Calculate total number of agents
total_agents = agents_per_gpu * gpu_count
print('total_agents:', total_agents)


# start sesstion
tmux('new-session -d -s sweep')

for gpu in range(gpu_count):
    for agent in range(agents_per_gpu):
        
        pane_name = f"{gpu}-{agent}"
        tmux(f'new-window -t sweep -n {pane_name}')
        command = f"CUDA_VISIBLE_DEVICES={gpu} XLA_PYTHON_CLIENT_PREALLOCATE=false wandb agent amacrutherford/{project}/{sweep_id}"
        
        tmux_shell(command)

