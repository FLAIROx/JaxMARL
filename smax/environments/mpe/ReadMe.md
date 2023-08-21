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
* simple Reference
* Simple Adversary

The implementations follow the PettingZoo code as closely as possible, including sharing variable names. There are occasional discrepncies between the PettingZoo code and docs, where this occurs we have followed the code. 

As our implementation closely follows the PettingZoo code, please refer to their documentation for general information on the environments.

## Action Space
Following the PettingZoo implementation, we allow for both discrete or continuous action spaces in all MPE envrionments. The environments use WHAT TODO by default.

## Citation
The environments were orginally described in the following work:
```
@article{mordatch2017emergence,
  title={Emergence of Grounded Compositional Language in Multi-Agent Populations},
  author={Mordatch, Igor and Abbeel, Pieter},
  journal={arXiv preprint arXiv:1703.04908},
  year={2017}
}
```