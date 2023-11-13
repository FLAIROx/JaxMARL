# Overcooked-JAX Environment

Overcooked-AI is a fully observable cooperative environment where two cooks (agents) must cooperate to prepare and serve onion soups. It is inspired by the popular videogame [Overcooked](https://ghosttowngames.com/overcooked/).

We implement all of the original Overcooked-AI layouts:
* Cramped Room
* Asymmetric Advantages
* Coordination Ring
* Forced Coordination
* Counter Circuit

We also provide a simple method for creating new layouts:
```python
custom_layout_grid = """
WWPWW
OA AO
W   W
WBWXW
"""
layout = layout_grid_to_dict(custom_layout_grid)
```
![grid](docs/cramped_room.gif)

The implementation aims to be as close as possible to the original Overcooked-AI environment, including dynamics, collision logic, and action and observation spaces.

#### A note on dynamics
In the original Overcooked-AI environment and in this JAX implementation, the pot starts cooking as soon as 3 onions are placed in the pot.
An update to Overcooked-AI has since changed the dynamics to require an additional pot interaction to start cooking. 
Updating the Overcooked-JAX to implement the new pot dynamics is on the roadmap and should be done by the end of 2023.

## Action Space
There are 6 possible actions, comprised of 4 movement actions (right, down, left, up), interact and no-op.

## Observation Space
The observations follow the featurization in the original Overcooked-AI environment, and is meant to be passed to a ConvNet.

Each observation is a sparse, (mostly) binary encoding of size `layout_height x layout_width x n_channels`, where `n_channels = 26`. 
For a detailed description of each channel, refer to the `get_obs(...)` method in [`overcooked.py`](overcooked.py).

## Get started
We provide an introduction on how to initialize, visualize and unroll a policy in the environment in `../../tutorials/overcooked_introduction.py`.

You can also try the environment yourself by running `python interactive.py`. Use the arrows to move both agents and the spacebar to interact.

## Visualization
We animate a collected set of state sequences.
```python
from jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer

state_seq = [state_t0, state_t1, ...]  # collected state sequences

viz =  OvercookedVisualizer()
viz.animate(state_seq, env.agent_view_size, filename='animation.gif')

```

## Limitations
Overcooked is an approachable and popular environment to study coordination, but has limitations, notably due to being fully observable.
For analysis on this topic, read more [here](https://arxiv.org/abs/2306.09309).

## Citation
The environment was orginally described in the following work:
```
@article{carroll2019utility,
  title={On the utility of learning about humans for human-ai coordination},
  author={Carroll, Micah and Shah, Rohin and Ho, Mark K and Griffiths, Tom and Seshia, Sanjit and Abbeel, Pieter and Dragan, Anca},
  journal={Advances in neural information processing systems},
  volume={32},
  year={2019}
}
```

## To Do
[] Clean up unused code (Randomised starts)

[] Update dynamics to match latest version of Overcooked-AI
