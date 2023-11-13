# Multi-Agent Brax Environments
This directory contains a subset of the multi-agent environments as described in the paper
[FACMAC: Factored Multi-Agent Centralised Policy Gradients](https://arxiv.org/abs/2003.06709).
The task descriptions are the same as in the implementation from [Gymnasium-Robotics](https://robotics.farama.org/envs/MaMuJoCo/).
These are multi-agent factorisations of MuJoCo tasks such that each agent controls a subset of the joints
and only observes the local state. 

Specifically, we include the following environments:

| Environment | Description |
| ----------- | ------------ | 
| `ant_4x2` | 4 agents, 2 joints each. One agent controls each leg. |
| `halfcheetah_6x1` | 6 agents, 1 joint each. One agent controls each joint. |
| `hopper_3x1` | 3 agents, 1 joint each. One agent controls each joint. |
| `humanoid_9\|8` | 2 agents, 9 and 8 joints. One agent controls the upper body, the other the lower body. |
| `walker2d_2x3` | 2 agents, 3 joints each. Factored into right and left leg. |

## Observation Space 
Each agent's observation vector is composed of the local state of the joints it controls, as well as the state of joints at distance 1 away in the body graph, and the state of the root body. State here refers to the position and velocity of the joint or body. All observations are continuous numbers in the range [-inf, inf].

## Action Spaces
Each agent's action space is the input torques to the joints it controls. All environments have continuous actions in the range [-1.0, 1.0], except for `humanoid_9|8` where the range is [-0.4, 0.4].


## Visualisation
To visualise a trajectory in a Jupyter notebook, given a list of states, you can use the following code snippet:
```python
from IPython.display import HTML
from brax.io import html

HTML(html.render(env.sys, [s.qp for s in state_history]))
```

## Differences to Gymnasium-Robotics MaMuJoCo
A notable difference to Gymansium-Robotics is that this JAX implementation currently fixes the observation distance to 1, whereas in the original implementation, it is a configurable parameter. This means that each agent has access to the observations of joints at distance 1 away from it in the body graph. We plan to make this a configurable parameter in a future update.
