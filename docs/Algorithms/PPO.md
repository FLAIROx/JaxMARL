# IPPO & MAPPO

# IPPO Baseline

Pure JAX IPPO implementation, based on the PureJaxRL PPO implementation.

## 🔎 Implementation Details
General features:

* Agents are controlled by a single network architecture (either FF or RNN).
* Parameters are shared between agents.

## 🚀 Usage

If you have cloned JaxMARL and are in the repository root, you can run the algorithms as scripts, e.g.
```bash
python baselines/IPPO/ippo_rnn_smax.py
```
Each file has a distinct config file which resides within [`config`](https://github.com/FLAIROx/JaxMARL/tree/main/baselines/IPPO/config).
The config file contains the IPPO hyperparameters, the environment's parameters and for some config files the `wandb` details (`wandb` is disabled by default).

# MAPPO Baseline

Pure JAX MAPPO implementation, based on the PureJaxRL PPO implementation.

## 🔎 Implementation Details
General features:

* Agents are controlled by a single network architecture (either FF or RNN).
* Parameters are shared between agents.
* Each script has a `WorldStateWrapper` which provides a global `"world_state"` observation.

## 🚀 Usage

If you have cloned JaxMARL and are in the repository root, you can run the algorithms as scripts, e.g.
```bash
python baselines/MAPPO/mappo_rnn_smax.py
```
Each file has a distinct config file which resides within [`config`](https://github.com/FLAIROx/JaxMARL/tree/main/baselines/MAPPO/config).
The config file contains the MAPPO hyperparameters, the environment's parameters and the `wandb` details (`wandb` is disabled by default).

