import jax
import yaml
import wandb
from os import system

ACTIVATE_VENV = '. path_to_your_virtualenv/bin/activate'


def tmux(command):
    system('tmux %s' % command)


def tmux_shell(command):
    tmux('send-keys "%s" "C-m"' % command)

# example: one tab with vim, other tab with two consoles (vertical split)
# with virtualenvs on the project, and a third tab with the server running

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

# # vim in project
# tmux('select-window -t 0')
# tmux_shell('vim')
# tmux('rename-window "vim"')

# # console in project
# tmux('new-window')
# tmux('select-window -t 1')
# tmux_shell('cd %s' % PROJECT_PATH)
# tmux_shell(ACTIVATE_VENV)
# tmux('rename-window "consola"')
# # second console as split
# tmux('split-window -v')
# tmux('select-pane -t 1')
# tmux_shell('cd %s' % PROJECT_PATH)
# tmux_shell(ACTIVATE_VENV)
# tmux('rename-window "consola"')

# # local server
# tmux('new-window')
# tmux('select-window -t 2')
# tmux_shell('cd %s' % PROJECT_PATH)
# tmux_shell(ACTIVATE_VENV)
# tmux_shell('python manage.py runserver')
# tmux('rename-window "server"')

# # go back to the first window
# tmux('select-window -t 0')
