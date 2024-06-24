# QLearning Baselines


Pure JAX implementations of:
* IQL (Independent Q-Learners)
* VDN (Value Decomposition Network)
* QMIX
* TransfQMix (Transformers for Leveraging the Graph Structure of MARL Problems)
* SHAQ (Incorporating Shapley Value Theory into Multi-Agent Q-Learning)

The first three are follow the original [Pymarl](https://github.com/oxwhirl/pymarl/blob/master/src/learners/q_learner.py) codebase while SHAQ follows the [paper code](https://github.com/hsvgbkhgbv/shapley-q-learning). PQN follows[purejaxql](https://github.com/mttga/purejaxql). 


```
❗
Standard algorithms (iql, vdn, qmix) support:
- MPE
- SMAX
- Overcooked

PQN supports:
- MPE
- SMAX
- Hanabi
- Overcooked
```

## ⚙️ Implementation Details

General features:

- Agents are controlled by a single RNN architecture.
- You can choose whether to share parameters between agents or not (not available on TransfQMix).
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
# vdn rnn in in mpe spread
python baselines/QLearning/vdn_rnn.py +alg=ql_rnn_mpe
# independent IQLs rnn in competetive simple_tag (predator-prey)
python baselines/QLearning/iql_rnn.py +alg=ql_rnn_mpe alg.ENV_NAME=MPE_simple_tag_v3
# QMix with SMAX
python baselines/QLearning/qmix.py +alg=qmix_smax +env=smax
# VDN overcooked with a different layout
python baselines/QLearning/vdn_cnn_overcooked.py +alg=ql_cnn_overcooked alg.ENV_KWARGS.LAYOUT=counter_circuit
# TransfQMix
python baselines/QLearning/transf_qmix.py +alg=transf_qmix_smax +env=smax

# pqn feed-forward with mpe
python baselines/QLearning/pqn_vdn_ff.py +alg=pqn_vdn_ff_mpe
# pqn feed-forward with hanabi
python baselines/QLearning/pqn_vdn_ff.py +alg=pqn_vdn_ff_hanabi
# pqn CNN with overcooked
python baselines/QLearning/pqn_vdn_cnn_overcooked.py +alg=pqn_vdn_cnn_overcooked
# pqn rnn with mpe
python baselines/QLearning/pqn_vdn_rnn.py +alg=pqn_vdn_cnn_overcooked
```

Notice that with Hydra, you can modify parameters on the go in this way:

```bash
# Run IQL without parameter sharing from the command line
python baselines/QLearning/iql_rnn.py +alg=ql_rnn_mpe alg.LR=0.001
```

**❗Note on Transformers**: TransfQMix currently supports only MPE_Spread and SMAX. You will need to wrap the observation vectors into matrices to use transformers in other environments. See: ```jaxmarl.wrappers.transformers```

## 🎯 Hyperparameter tuning

Please refer to the ```tune``` function in the [transf_qmix.py](transf_qmix.py) script for an example of hyperparameter tuning using WANDB. 