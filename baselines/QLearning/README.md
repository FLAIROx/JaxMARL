# QLearning Baselines


```diff
- these implementations were tested with python 3.8 and jax 0.4.8 -> 0.4.11
- With jax 0.4.13 you could experience a degradation of performances.
```

Pure-Jax mplementation of **IQL** (Independent Q-Learners), **VDN** (Value Decomposition Network) and **QMIX**. The implementation follows the original [Pymarl](https://github.com/oxwhirl/pymarl/blob/master/src/learners/q_learner.py).


### Implementation remarks

General features:

- Agents are controlled by a single RNN architecture.
- Works also with non-homogenous agents (different obs/action spaces).
- Experience replay is a simple buffer with uniform sampling.
- Uses Double Q-Learning with a target agent network (hard-updated).
- You can choose between the DDQN loss as the original pymarl or the TD_Lambda loss as pymarl2.
- Adam optimizer is used instead (not RMSPROP as in pymarl).
- The environment is reset at the end of each episode.
- Currently, the `last_action` feature is not included in the agents' observations.

Three versions of the algorithms are present:

1. Without parameters sharing (iql_ns.py, vdn_ns.py, qmix_ns.py): each agents has its own parameters, and all the agents operations are vmapped in respect to them.
2. With parameters sharing (iql_ps.py, vdn_ps.py, qmix_ps.py): the same set of parameters are assigned to all agent, and all the agents operations are computed in a single batch.
3. With parameters sharing using pretrained agents (iql_ps_pretrained.py, vdn_ps_pretrained.py, qmix_ps_pretrained.py): these scripts allow to use some pretrained networks to control some of the environment agents.

All the algs use the `CTRolloutManager` env wrapper (found it in utils.py) which is used to:

- Batchify the step and reset functions to run parallel environments.
- Add a global observation (`obs["__all__"]`) and a global reward (`rewards["__all__"]`) to the returns of `env.step`, for centralized training.
- Preprocess and uniform the observation vectors (flat, pad, add additional features like id one-hot encoding, etc.)

Please modify this wrapper for you needs.

### Usage

Often is useful to run these scripts in a notebook or in another script, therefore they are designed as a module. If you cloned the SMAX repository and you are in the repo root, you can do something like:

```python
from smax import make
from baselines.QLearning.iql_ps import make_train as iql_ps

env = make("MPE_simple_spread_v3")

config = {
        "NUM_ENVS": 8,
        "BUFFER_SIZE": 5000,
        "BUFFER_BATCH_SIZE": 32,
        "TOTAL_TIMESTEPS": 2e6,
        "AGENT_HIDDEN_DIM": 64,
        "AGENT_INIT_SCALE": 2.0,
        "EPSILON_START": 1.0,
        "EPSILON_FINISH": 0.05,
        "EPSILON_ANNEAL_TIME": 1e5,
        "MAX_GRAD_NORM": 25,
        "TARGET_UPDATE_INTERVAL": 200 ,
        "LR": 5e-3,
        "EPS_ADAM": 1e-3,
        "TD_LAMBDA_LOSS": True,
        "TD_LAMBDA": 0.6,
        "GAMMA": 0.99,
        "DEBUG": False,
        "NUM_TEST_EPISODES": 32,
        "TEST_INTERVAL": 5e4,
        "VERBOSE": True,
        "WANDB_ONLINE_REPORT": False,
        "SEED": 30,
        "NUM_SEEDS": 10,
    }

rng = jax.random.PRNGKey(42)
train_vjit = jax.jit((iql_ps(config, env))
outs_iql = train_vjit(rng)
```

If you wish to run the code as a script, you can still do that by running it as a module:

```bash
python -m baselines.QLearning.iql_ps
```

### TODO
- IQL NS checkpoint for simple tag