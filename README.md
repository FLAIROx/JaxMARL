# Scalable MARL in JAX (SMAX)

Welcome to the work-in-progress Scalable MARL in Jax (SMAX) library. We build heavily off Gymnax and take API inspiration from PettingZoo.

## Basic `SMAX` API  Usage
```python 
import jax
from smax import make

rng = jax.random.PRNGKey(0)
rng, key_reset, key_act, key_step = jax.random.split(rng, 4)

# Initialise environment.
env = make('MPE_simple_world_comm_v3')

# Reset the environment.
obs, state = env.reset(key_reset)

# Sample random actions.
key_act = jax.random.split(key_a, env.num_agents)
actions = {agent: env.action_space(agent).sample(key_act[i]) for i, agent in enumerate(env.agents)}

# Perform the step transition.
n_obs, n_state, reward, done, infos = env.step(key_step, state, action)
```

## Installation
Using Conda, run the following commands. You must ensure you install the correct JAX version, more information can be found [here](https://github.com/google/jax#installation)
```
conda create -n smax python=3.8
conda activate smax

# NOTE this installs jax for CUDA 12
# if you have a different CUDA version or no GPU, see https://github.com/google/jax#installation
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

conda install -c conda-forge flax chex tqdm wandb dotmap
pip install gymnax evosax pettingzoo brax==0.0.16
pip install -e .

# to run MPE tests, you will also need pygame
# pip install pygame
```

## Contributing 
Create a branch and send a merge request once your environment passes tests that show consistency with the existing implementation.
