# Coin Game

This directory contains an implementation of the Coin Game environment presented in [Model-Free Opponent Shaping (Lu et al.)](https://arxiv.org/abs/2205.01447). The original description and usage of the environment is from [Maintaining cooperation in complex social dilemmas using deep reinforcement learning (Lerer et al.)](https://arxiv.org/abs/1707.01068), and [Learning with Opponent-Learning Awareness (Foerster et al.)](https://arxiv.org/abs/1709.04326) is the first to popularize its use for opponent shaping. A description from Model-Free Opponent Shaping:

```
The Coin Game is a multi-agent grid-world environment that simulates social dilemmas like the IPD but with high dimensional dynamic states. First proposed by Lerer & Peysakhovich (2017), the game consists of two players, labeled red and blue respectively, who are tasked with picking up coins, also labeled red and blue respectively, in a 3x3 grid. If a player picks up any coin by moving into the same position as the coin, they receive a reward of +1.  However, if they pick up a coin of the other player’s color, the other player receives a reward of −2. Thus, if both agents play greedily and pick up every coin, the expected reward for both agents is 0.
```

