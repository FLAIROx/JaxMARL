# Multi-Agent MuJoCo Environments
This directory contains a subset of the multi-agent MuJoCo environments as described in the paper
[FACMAC: Factored Multi-Agent Centralised Policy Gradients](https://arxiv.org/abs/2003.06709).
The task descriptions are the same as in the implementation from [Gymnasium-Robotics](https://robotics.farama.org/envs/MaMuJoCo/).
These are multi-agent factorisations of MuJoCo tasks such that each agent controls a subset of the rotors
and only observes the local state. 

Specifically, we include the following environments:

| Environment | Description |
| ----------- | ------------ | 
| `ant_4x2` | 4 agents, 2 rotors each. One agent controls each leg. |
| `halfcheetah_6x1` | 6 agents, 1 rotor each. One agent controls each joint. |
| `hopper_3x1` | 3 agents, 1 rotor each. One agent controls each joint. |
| `humanoid_9\|8` | 2 agents, 9 and 8 rotors. One agent controls the upper body, the other the lower body. |
| `walker2d_2x3` | 2 agents, 3 rotors each. Factored into right and left leg. |

## Observation and Action Spaces
Each agent's observations are composed of the local state of the rotors it controls, as well as the state of joints at distance 1 away in the kinematic tree, and the state of the root body. State here refers to the position and velocity of the joint or rotor. 

Each agent's action space is the torques of the rotors it controls.
