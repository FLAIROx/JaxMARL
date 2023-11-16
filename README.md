<h1 align="center">JaxMARL</h1>

<p align="center">
       <a href="https://pypi.python.org/pypi/jaxmarl">
        <img src="https://img.shields.io/pypi/pyversions/jaxmarl.svg" /></a>
       <a href= "https://badge.fury.io/py/jaxmarl">
        <img src="https://badge.fury.io/py/jaxmarl.svg" /></a>
       <a href= "https://github.com/FLAIROx/JaxMARL/blob/main/LICENSE">
        <img src="https://img.shields.io/badge/license-Apache2.0-blue.svg" /></a>
</p>

[**Installation**](#install) | [**Quick Start**](#start) | [**Environments**](#environments) | [**Algorithms**](#algorithms) | [**Citation**](#cite)
---

<div class="collage">
    <div class="row" align="centre">
        <img src="https://github.com/FLAIROx/JaxMARL/blob/main/docs/imgs/cramped_room.gif?raw=true" alt="Overcooked" width="20%">
        <img src="https://github.com/FLAIROx/JaxMARL/blob/main/docs/imgs/qmix_MPE_simple_tag_v3.gif?raw=true" alt="MPE" width="20%">
        <img src="https://github.com/FLAIROx/JaxMARL/blob/main/docs/imgs/storm.gif?raw=true" alt="STORM" width="20%">
        <img src="https://github.com/FLAIROx/JaxMARL/blob/main/docs/imgs/smax.gif?raw=true" alt="SMAX" width="20%">
    </div>
</div>

## Multi-Agent Reinforcement Learning in JAX

JaxMARL combines ease-of-use with GPU enabled efficiency, and supports a wide range of commonly used MARL environments as well as popular baseline algorithms. Our aim is for one library that enables thorough evaluation of MARL methods across a wide range of tasks and against relevant baselines. We also introduce SMAX, a vectorised, simplifed version of the popular StarCraft Multi-Agent Challenge, which removes the need to run the StarCraft II game engine. 

For more details, take a look at our blog post or this notebook walks through the basic usage. LINKS TODO

<h2 name="environments" id="environments">Environments 🌍 </h2>

| Environment | Reference | README | Summary |
| --- | --- | --- | --- |
| 🔴 MPE | [Paper](https://arxiv.org/abs/1706.02275) | [Source](https://github.com/FLAIROx/JaxMARL/tree/main/jaxmarl/environments/mpe) | Communication orientated tasks in a multi-agent particle world
| 🍲 Overcooked | [Paper](https://arxiv.org/abs/1910.05789) | [Source](https://github.com/FLAIROx/JaxMARL/tree/main/jaxmarl/environments/overcooked) | Fully-cooperative human-AI coordination tasks based on the homonyms video game | 
| 🦾 Multi-Agent Brax | [Paper](https://arxiv.org/abs/2003.06709) | [Source](https://github.com/FLAIROx/JaxMARL/tree/main/jaxmarl/environments/mabrax) | Continuous multi-agent robotic control based on Brax, analagous to Multi-Agent MuJoCo |
| 🎆 Hanabi | [Paper](https://arxiv.org/abs/1902.00506) | [Source](https://github.com/FLAIROx/JaxMARL/tree/main/jaxmarl/environments/hanabi) | Fully-cooperative partially-observable multiplayer card game |
| 👾 SMAX | Novel | [Source](https://github.com/FLAIROx/JaxMARL/tree/main/jaxmarl/environments/smax) | Simplifed cooperative StarCraft micro-management environment |
| 🧮 STORM: Spatial-Temporal Representations of Matrix Games | [Paper](https://openreview.net/forum?id=54F8woU8vhq) | [Source](https://github.com/FLAIROx/JaxMARL/tree/main/jaxmarl/environments/storm) | Matrix games represented as grid world scenarios
| 🪙 Coin Game | [Paper](https://arxiv.org/abs/1802.09640) | [Source](https://github.com/FLAIROx/JaxMARL/tree/main/jaxmarl/environments/coin_game) | Two-player grid world environment which emulates social dilemmas
| 💡 Switch Riddle | [Paper](https://proceedings.neurips.cc/paper_files/paper/2016/hash/c7635bfd99248a2cdef8249ef7bfbef4-Abstract.html) | [Source](https://github.com/FLAIROx/JaxMARL/tree/main/jaxmarl/environments/switch_riddle) | Simple cooperative communication game included for debugging

 
<h2 name="algorithms" id="algorithms">Baseline Algorithms 🦉 </h2>

We follow CleanRL's philosophy of providing single file implementations which can be found within the `baselines` directory.

| Algorithm | Reference | README | 
| --- | --- | --- | 
| IPPO | [Paper](https://arxiv.org/pdf/2011.09533.pdf) | [Source](https://github.com/FLAIROx/JaxMARL/tree/main/baselines/IPPO) | 
| MAPPO | [Paper](https://arxiv.org/abs/2103.01955) | [Source](https://github.com/FLAIROx/JaxMARL/tree/main/baselines/MAPPO) | 
| IQL | [Paper](https://arxiv.org/abs/1312.5602v1) | [Source](https://github.com/FLAIROx/JaxMARL/tree/main/baselines/QLearning) | DQN 
| VDN | [Paper](https://arxiv.org/abs/1706.05296)  | [Source](https://github.com/FLAIROx/JaxMARL/tree/main/baselines/QLearning) |
| QMIX | [Paper](https://arxiv.org/abs/1803.11485) | [Source](https://github.com/FLAIROx/JaxMARL/tree/main/baselines/QLearning) |

<h2 name="install" id="install">Installation 🧗 </h2>

Before installing, ensure you have the correct [JAX version](https://github.com/google/jax#installation) for your hardware accelerator. JaxMARL can then be installed directly from PyPi:

```
pip install jaxmarl 
```
We have tested JaxMARL on Python 3.8 and 3.9. To run our test scripts, some additional dependencies are required (for comparisons against existing implementations), these can be installed with:
```
pip install jaxmarl[dev]
```

<h2 name="start" id="start">Quick Start 🚀 </h2>

We take inspiration from the [PettingZoo](https://github.com/Farama-Foundation/PettingZoo) and [Gymnax](https://github.com/RobertTLange/gymnax) interfaces. You can try out training an agent on XX in this Colab TODO. Further introduction scripts can be found [here](https://github.com/FLAIROx/JaxMARL/tree/main/jaxmarl/tutorials).

### Basic JaxMARL API  Usage 🖥️

Actions, observations, rewards and done values are passed as dictionaries keyed by agent name, allowing for differing action and observation spaces. The done dictionary contains an additional `"__all__"` key, specifying whether the episode has ended. We follow a parallel structure, with each agent passing an action at each timestep. For ascyhronous games, such as Hanabi, a dummy action is passed for agents not acting at a given timestep.

```python 
import jax
from jaxmarl import make

key = jax.random.PRNGKey(0)
key, key_reset, key_act, key_step = jax.random.split(rng, 4)

# Initialise environment.
env = make('MPE_simple_world_comm_v3')

# Reset the environment.
obs, state = env.reset(key_reset)

# Sample random actions.
key_act = jax.random.split(key_act, env.num_agents)
actions = {agent: env.action_space(agent).sample(key_act[i]) for i, agent in enumerate(env.agents)}

# Perform the step transition.
obs, state, reward, done, infos = env.step(key_step, state, actions)
```

## Contributing 🔨
Please contribute! Please take a look at our [contributing guide](https://github.com/FLAIROx/JaxMARL/blob/main/CONTRIBUTING.md) for how to add an environment/algorithm or submit a bug report.

<h2 name="cite" id="cite">Citing JaxMARL 📜 </h2>
If you use JaxMARL in your work, please cite us as follows:

```
TODO
```

## See Also 🙌
There are a number of other libraries which inspired this work, we encourage you to take a look!

JAX-native algorithms:
- [Mava](https://github.com/instadeepai/Mava): JAX implementations of IPPO and MAPPO, two popular MARL algorithms.
- [PureJaxRL](https://github.com/luchris429/purejaxrl): JAX implementation of PPO, and demonstration of end-to-end JAX-based RL training.

JAX-native envrionments:
- [Gymnax](https://github.com/RobertTLange/gymnax): Implementations of classic RL tasks including classic control, bsuite and MinAtar.
- [Jumanji](https://github.com/instadeepai/jumanji): A diverse set of environments ranging from simple games to NP-hard combinatoral problems.
- [Pgx](https://github.com/sotetsuk/pgx): JAX implementations of classic board games, such as Chess, Go and Shogi.
- [Brax](https://github.com/google/brax): A fully differentiable physics engine written in JAX, features continuous control tasks.
