# IPPO Baseline

Pure JAX IPPO implementation, based on the PureJaxRL PPO implementation.

## 🔎 Implementation Details
General features:
* Agents are controlled by a single network architecture (either FF or RNN).
* Parameters are shared between agents.

## 🚀 Usage

If you have cloned JaxMARL and are in the repository root, you can run the algorithms as scripts, e.g.
```
python baselines/IPPO/ippo_rnn_smax.py
```
Each file has a distinct config file which resides within [`config`](https://github.com/FLAIROx/JaxMARL/tree/main/baselines/IPPO/config).
The config file contains the IPPO hyperparameters, the environment's parameters and for some config files the `wandb` details (`wandb` is disabled by default).
