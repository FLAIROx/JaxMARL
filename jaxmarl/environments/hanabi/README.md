# Hanabi-JAX Environment

This directory contains a MARL environment for the cooperative card game, Hanabi, implemented in JAX. It is inspired by the popular [Hanabi Learning Environment (HLE)](https://arxiv.org/pdf/1902.00506.pdf), but intended to be simpler to integrate and run with the growing ecosystem of JAX implemented RL research pipelines. 


## Action Space
Hanabi is a turn-based game. The current player can choose to discard or play any of the cards in their hand, or hint a colour or rank to any one of their teammates.

## Observation Space
The observations closely follow the featurization in the HLE. Each observation is comprised of 658 features:

* **Hands (127)**: information about the visible hands.
  * other player hand: 125 
    * card 0: 25,
    * card 1: 25
    * card 2: 25
    * card 3: 25
    * card 4: 25
  * Hands missing card: 2 (one-hot)

* **Board (76)**: encoding of the public information visible in the board.
  * Deck: 40, thermometer 
  * Fireworks: 25, one-hot
  * Info Tokens: 8, thermometer
  * ife Tokens: 3, thermometer

* **Discards (50)**: encoding of the cards in the discard pile.
  * Colour R: 10 bits for each card
  * Colour Y: 10 bits for each card
  * Colour G: 10 bits for each card
  * Colour W: 10 bits for each card
  * Colour B: 10 bits for each card

* **Last Action (55)**: encoding of the last move of the previous player.
  * Acting player index, relative to yourself: 2, one-hot
  * MoveType: 4, one-hot
  * Target player index, relative to acting player: 2, one-hot
  * Color revealed: 5, one-hot
  * Rank revealed: 5, one-hot
  * Reveal outcome 5 bits, each bit is 1 if the card was hinted at
  * Position played/discarded: 5, one-hot
  * Card played/discarded 25, one-hot
  * Card played scored: 1
  * Card played added info token: 1

* **V0 belief (350)**: trivially-computed probability of being a specific car (given the played-discarded cards and the hints given), for each card of each player.
  * Possible Card (for each card): 25 (* 10)
  * Colour hinted (for each card): 5 (* 10)
  * Rank hinted (for each card): 5 (* 10)

## Pretrained Models

We make available to use some pretrained models. For example you can use a jax conversion of the original R2D2 OBL model in this way:

1. Download the models from Hugginface: ```git clone https://huggingface.co/mttga/obl-r2d2-flax``` (ensure to have git lfs installed). You can also use the script: bash jaxmarl/environments/hanabi/models/download_r2d2_obl.sh
2. Load the parameters, import the agent wrapper and use it with JaxMarl Hanabi:

```python
import jax
from jax import numpy as jnp
from jaxmarl import make
from jaxmarl.wrappers.baselines import load_params
from jaxmarl.environments.hanabi.pre_trained import OBLAgentR2D2

weight_file = "jaxmarl/environments/hanabi/pre_trained/obl-r2d2-flax/icml_OBL1/OFF_BELIEF1_SHUFFLE_COLOR0_BZA0_BELIEF_a.safetensors"
params = load_params(weight_file)

agent = OBLAgentR2D2()
agent_carry = agent.initialize_carry(jax.random.PRNGKey(0), batch_dims=(2,))

rng = jax.random.PRNGKey(0)
env = make('hanabi')
obs, env_state = env.reset(rng)
env.render(env_state)

batchify = lambda x: jnp.stack([x[agent] for agent in env.agents])
unbatchify = lambda x: {agent:x[i] for i, agent in enumerate(env.agents)}

agent_input = (
    batchify(obs),
    batchify(env.get_legal_moves(env_state))
)
agent_carry, actions = agent.greedy_act(params, agent_carry, agent_input)
actions = unbatchify(actions)

obs, env_state, rewards, done, info = env.step(rng, env_state, actions)

print('actions:', {agent:env.action_encoding[int(a)] for agent, a in actions.items()})
env.render(env_state)
```

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
