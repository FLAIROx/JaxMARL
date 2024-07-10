# QLearning Baselines

Pure JAX implementations of:
* PQN-VDN (Prallelised Q-Network)
* IQL (Independent Q-Learners)
* VDN (Value Decomposition Network)
* QMIX
* TransfQMix (Transformers for Leveraging the Graph Structure of MARL Problems)
* SHAQ (Incorporating Shapley Value Theory into Multi-Agent Q-Learning)

PQN implementation follows [purejaxql](https://github.com/mttga/purejaxql). IQL, VDN and QMix follow the original [Pymarl](https://github.com/oxwhirl/pymarl/blob/master/src/learners/q_learner.py) codebase while SHAQ follows the [paper code](https://github.com/hsvgbkhgbv/shapley-q-learning). 


Standard algorithms (iql, vdn, qmix) support:
- MPE
- SMAX
- Overcooked (qmix not supported)

PQN-VDN supports:
- MPE
- SMAX
- Hanabi
- Overcooked

**At the moment, PQN-VDN should be the most performant baseline for Q-Learning in terms of returns and training speed.**

❗ TransfQMix and Shaq still use an old implementation of the scripts and need refactoring to match the new format. 


## ⚙️ Implementation Details

All the algorithms take advantage of the `CTRolloutManager` environment wrapper (found in `jaxmarl.wrappers.baselines`), which is used to:

- Batchify the step and reset functions to run parallel environments.
- Add a global observation (`obs["__all__"]`) and a global reward (`rewards["__all__"]`) to the returns of `env.step` for centralized training.
- Preprocess and uniform the observation vectors (flatten, pad, add additional features like id one-hot encoding, etc.).

You might want to modify this wrapper for your needs.

## 🚀 Usage

If you have cloned JaxMARL and you are in the repository root, you can run the algorithms as scripts. You will need to specify which parameter configurations will be loaded by Hydra by choosing them (or adding yours) in the config folder. Below are some examples:

```bash
# vdn rnn in in mpe spread
python baselines/QLearning/vdn_rnn.py +alg=ql_rnn_mpe
# independent IQL rnn in competetive simple_tag (predator-prey)
python baselines/QLearning/iql_rnn.py +alg=ql_rnn_mpe alg.ENV_NAME=MPE_simple_tag_v3
# QMix with SMAX
python baselines/QLearning/qmix_rnn.py +alg=ql_rnn_smax
# VDN overcooked
python baselines/QLearning/vdn_cnn_overcooked.py +alg=ql_cnn_overcooked alg.ENV_KWARGS.LAYOUT=counter_circuit
# TransfQMix
python baselines/QLearning/transf_qmix.py +alg=transf_qmix_smax

# pqn feed-forward in MPE
python baselines/QLearning/pqn_vdn_ff.py +alg=pqn_vdn_ff_mpe
# pqn feed-forward in hanabi
python baselines/QLearning/pqn_vdn_ff.py +alg=pqn_vdn_ff_hanabi
# pqn CNN in overcooked
python baselines/QLearning/pqn_vdn_cnn_overcooked.py +alg=pqn_vdn_cnn_overcooked
# pqn with RNN in SMAX
python baselines/QLearning/pqn_vdn_rnn.py +alg=pqn_vdn_rnn_smax
```

Notice that with Hydra, you can modify parameters on the go in this way:

```bash
# change learning rate
python baselines/QLearning/iql_rnn.py +alg=ql_rnn_mpe alg.LR=0.001
# change overcooked layout
python baselines/QLearning/pqn_vdn_cnn_overcooked.py +alg=pqn_vdn_cnn_overcooked alg.ENV_KWARGS.LAYOUT=counter_circuit
# change smax map
python baselines/QLearning/pqn_vdn_rnn.py +alg=pqn_vdn_rnn_smax alg.MAP_NAME=5m_vs_6m
```

Take a look at [`config.yaml`](./config/config.yaml) for the default configuration when running these scripts. There you can choose how many seeds to vmap and you can setup WANDB. 

**❗Note on Transformers**: TransfQMix currently supports only MPE_Spread and SMAX. You will need to wrap the observation vectors into matrices to use transformers in other environments. See: ```jaxmarl.wrappers.transformers```

## 🎯 Hyperparameter tuning

All the scripts include a tune function to perform hyperparameter tuning. To use it, set `"HYP_TUNE": True` in the `config.yaml` and set the hyperparameters spaces in the tune function. For more information, check [wandb documentation](https://docs.wandb.ai/guides/sweeps).