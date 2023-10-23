# MPE Environments

Multi Particle Environments (MPE) are a set of communication oriented environment where particle agents can (sometimes) move, communicate, see each other, push each other around, and interact with fixed landmarks.

We implement all of the [PettingZoo MPE Environments](https://pettingzoo.farama.org/environments/mpe/):
* Simple
* Simple Push
* Simple Spread
* Simple Crypto
* Simple Speaker Listener
* Simple World Comm
* Simple Tag
* Simple Reference
* Simple Adversary

The implementations follow the PettingZoo code as closely as possible, including sharing variable names. There are occasional discrepancies between the PettingZoo code and docs, where this occurs we have followed the code. 

As our implementation closely follows the PettingZoo code, please refer to their documentation for general information on the environments.

We additionally include a fully cooperative variant of Simple Tag, first used to evaluate FACMAC. In this environmnet, a number of agents attempt to tag a number of prey, where the prey are controlled by a heuristic AI.

## Action Space
Following the PettingZoo implementation, we allow for both discrete or continuous action spaces in all MPE envrionments. The environments use discrete actions by default.

## Visualisation
Check the example `mpe_introduction.py` file in the tutorials folder for an introduction to our implementation of the MPE environments, including an example visualisation. We animate the environment after the state transitions have been collected as follows:

```python
from smax import make
from smax.environments.mpe import MPEVisualizer

env = make("MPE_simple_v3")

# state_seq is a list of the jax env states passed to the step function
# i.e. [state_t0, state_t1, ...]
viz = MPEVisualizer(env, state_seq)
viz.animate(view=True)  # can also save the animiation as a .gif file
```

## Citation
MPE was orginally described in the following work:
```
@article{mordatch2017emergence,
  title={Emergence of Grounded Compositional Language in Multi-Agent Populations},
  author={Mordatch, Igor and Abbeel, Pieter},
  journal={arXiv preprint arXiv:1703.04908},
  year={2017}
}
```
The fully cooperative Simple Tag variant was proposed by:
```
@article{peng2021facmac,
  title={Facmac: Factored multi-agent centralised policy gradients},
  author={Peng, Bei and Rashid, Tabish and Schroeder de Witt, Christian and Kamienny, Pierre-Alexandre and Torr, Philip and B{\"o}hmer, Wendelin and Whiteson, Shimon},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={12208--12221},
  year={2021}
}
```

