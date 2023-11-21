# Hanabi-JAX Environment

This directory contains a MARL environment for the cooperative card game, Hanabi, implemented in JAX. It is inspired by the popular [Hanabi Learning Environment (HLE)](https://arxiv.org/pdf/1902.00506.pdf), but intended to be simpler to integrate and run with the growing ecosystem of JAX implemented RL research pipelines. 

#### A note on tuning 
The performance of IPPO on Hanabi, as implemented in this repo, is currently marginally lower than the reported [SoTA result for IPPO](https://arxiv.org/pdf/2103.01955.pdf). They run a very extensive hyperparameter sweep and conducting similarly comprehensive tuning of the JAX implemnation is on the near-term agenda.

## Action Space
Hanabi is a turn-based game. The current player can choose to discard or play any of the cards in their hand, or hint a colour or rank to any one of their teammates.

## Observation Space
The observations closely follow the featurization in the HLE.

Each observation is comprised of:
- card knowledge (binary encoding of implicit card knowledge); size `(hand_size * num_colors * num_ranks)`
- color and rank hints (binary encoding of explicit hints made about player's hand); size `(hand_size * (num_colors + num_ranks)`
- fireworks (thermometer encoded); size `(num_colors * num_ranks)`
- info tokens (thermometer encoded); size `max_info_tokens`
- life tokens (thermometer encoded); size `max_life_tokens`
- last moves (one-hot encoding of most recent move of each player); size `(num_players * num_moves)`
- current player (one-hot encoding); size `num_players`
- discard pile (one-hot encodings of discarded cards); size `(num_cards_of_color * num_colors * num_colors * num_ranks)
- remaining deck size (thermometer encoded); size `(num_cards_of_color * num_colors)`

## Citation
The environment was orginally described in the following work:
```
@article{bard2019hanabi,
  title={The Hanabi Challenge: A New Frontier for AI Research},
  author={Bard, Nolan and Foerster, Jakob N. and Chandar, Sarath and Burch, Neil and Lactot, Marc and Song,    H. Francis and Parisotto, Emilio and Dumoulin, Vincent and Moitra, Subhodeep and Hughes, Edward and          Dunning, Ian and Mourad, Shibl and Larochelle, Hugo and Bellemare, Marc G. and Bowling},
  journal={Artificial Intelligence Journal},
  year={2019}
}
```

## To Do
- [ ] Algorithm tuning