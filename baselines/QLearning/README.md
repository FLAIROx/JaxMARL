# QLearning Baselines


Pure JAX implementations of:
* IQL (Independent Q-Learners)
* VDN (Value Decomposition Network)
* QMIX
* SHAQ (Incorporating Shapley Value Theory into Multi-Agent Q-Learning)

The first three are follow the original [Pymarl](https://github.com/oxwhirl/pymarl/blob/master/src/learners/q_learner.py) codebase while SHAQ follows the [paper code](https://github.com/hsvgbkhgbv/shapley-q-learning)

```
⚠️ The implementations were tested with Python 3.9 and Jax 0.4.11. 
With Jax 0.4.13, you could experience a degradation of performance. 
```

We use [`flashbax`](https://github.com/instadeepai/flashbax) to provide our replay buffers, this requires Python 3.9 and the dependency can be installed with:
``` 
pip install -r requirements/requirements-qlearning.txt 
```

```
❗The implementations were tested in the following environments:
- MPE
- SMAX
- Hanabi
```

## 🔎 Implementation Details

General features:

- Agents are controlled by a single RNN architecture.
- You can choose whether to share parameters between agents or not.
- Works also with non-homogeneous agents (different observation/action spaces).
- Experience replay is a simple buffer with uniform sampling.
- Uses Double Q-Learning with a target agent network (hard-updated).
- You can select between TD Loss (pymarl2) or DDQN loss (pymarl).
- Adam optimizer is used instead of RMSPROP.
- The environment is reset at the end of each episode.
- Trained with a team reward (reward['__all__']).
- At the moment, last_actions are not included in the agents' observations.

All the algorithms take advantage of the `CTRolloutManager` environment wrapper (found in `jaxmarl.wrappers.baselines`), which is used to:

- Batchify the step and reset functions to run parallel environments.
- Add a global observation (`obs["__all__"]`) and a global reward (`rewards["__all__"]`) to the returns of `env.step` for centralized training.
- Preprocess and uniform the observation vectors (flatten, pad, add additional features like id one-hot encoding, etc.).

Please modify this wrapper for your needs.

## 🚀 Usage

If you have cloned JaxMARL and you are in the repository root, you can run the algorithms as scripts. You will need to specify which parameter configurations will be loaded by Hydra by choosing them (or adding yours) in the config folder. Below are some examples:

```bash
# IQL with MPE speaker-listener
python baselines/QLearning/iql.py +alg=iql_mpe +env=mpe_speaker_listener
# VDN with MPE spread
python baselines/QLearning/vdn.py +alg=vdn_mpe +env=mpe_spread
# QMix with SMAX
python baselines/QLearning/qmix.py +alg=qmix_smax +env=smax
# QMix with hanabi
python baselines/QLearning/qmix.py +alg=qmix_hanabi +env=hanabi
# QMix against pretrained agents
python baselines/QLearning/qmix_pretrained.py +alg=qmix_mpe +env=mpe_tag_pretrained
```

Notice that with Hydra, you can modify parameters on the go in this way:

```bash
# Run IQL without parameter sharing from the command line
python baselines/QLearning/iql.py +alg=iql_mpe +env=mpe_spread alg.PARAMETERS_SHARING=False
```

It is often useful to run these scripts manually in a notebook or in another script.

```python
from jaxmarl import make
from baselines.QLearning.qmix import make_train

env = make("MPE_simple_spread_v3")

config = {
    "NUM_ENVS": 8,
    "BUFFER_SIZE": 5000,
    "BUFFER_BATCH_SIZE": 32,
    "TOTAL_TIMESTEPS": 2050000,
    "AGENT_HIDDEN_DIM": 64,
    "AGENT_INIT_SCALE": 2.0,
    "PARAMETERS_SHARING": True,
    "EPSILON_START": 1.0,
    "EPSILON_FINISH": 0.05,
    "EPSILON_ANNEAL_TIME": 100000,
    "MIXER_EMBEDDING_DIM": 32,
    "MIXER_HYPERNET_HIDDEN_DIM": 64,
    "MIXER_INIT_SCALE": 0.00001,
    "MAX_GRAD_NORM": 25,
    "TARGET_UPDATE_INTERVAL": 200,
    "LR": 0.005,
    "LR_LINEAR_DECAY": True,
    "EPS_ADAM": 0.001,
    "WEIGHT_DECAY_ADAM": 0.00001,
    "TD_LAMBDA_LOSS": True,
    "TD_LAMBDA": 0.6,
    "GAMMA": 0.9,
    "VERBOSE": False,
    "WANDB_ONLINE_REPORT": False,
    "NUM_TEST_EPISODES": 32,
    "TEST_INTERVAL": 50000,
}

rng = jax.random.PRNGKey(42)
train_vjit = jax.jit(make_train(config, env))
outs = train_vjit(rng)
```