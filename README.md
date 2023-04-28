# SMAX ..or MultiAgentGymnax

Welcome to the work-in-progress Scalable MARL in Jax (SMAX) library. We build heavily off Gymnax and take API inspiration from PettingZoo.

## Basic `SMAX` API  Usage
```python 
import jax
from multiagentgymnax import make

rng = jax.random.PRNGKey(0)
rng, key_reset, key_act, key_step = jax.random.split(rng, 4)

# Initialise environment.
env = make('simple_world_comm_v2')

# Reset the environment.
obs, state = env.reset(key_reset, env_params)

# Sample random actions.
key_act = jax.random.split(key_a, env.num_agents)
actions = {agent: env.action_space(agent).sample(key_act[i]) for i, agent in enumerate(env.agents)}

# Perform the step transition.
n_obs, n_state, reward, done, infos = env.step(key_step, state, action)
```
An example environment loop can be found in `introduction.py`

## Installation
Using Conda, run the following commands (which can also be found in `env_commands`). You must ensure you install the correct JAX version, more information can be found [here](https://github.com/google/jax#installation)
```
conda create -n majax python=3.8

# NOTE this installs jax for CUDA 12
# if you have a different CUDA version or no GPU, see https://github.com/google/jax#installation
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

conda install -c conda-forge flax chex tqdm wandb dotmap
pip install gymnax evosax pettingzoo
pip install -e .

# to run MPE tests, you will also need pygame
# pip install pygame
```
