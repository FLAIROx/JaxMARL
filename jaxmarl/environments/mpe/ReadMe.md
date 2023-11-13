# MPE Environments

Multi Particle Environments (MPE) are a set of communication oriented environment where particle agents can (sometimes) move, communicate, see each other, push each other around, and interact with fixed landmarks. We implement all of the [PettingZoo MPE Environments](https://pettingzoo.farama.org/environments/mpe/).


<div class="collage">
  <div class="row" align="left">
    <img src="docs/qmix_MPE_simple_tag_v3.gif" alt="MPE Simple Tag" width="30%"/>
    <img src="docs/vdn_MPE_simple_spread_v3.gif" alt="MPE Simple Spread" width="30%"/>
    <img src="docs/qmix_MPE_simple_speaker_listener_v4.gif" alt="MPE Speaker Listener" width="30%">
  </div>
</div>



| Envrionment  | JaxMARL Registry Name  |
|---|---|
| Simple  | `MPE_simple_v3` |
| Simple Push  | `MPE_simple_push_v3`  |
| Simple Spread  |  `MPE_simple_spread_v3` |
| Simple Crypto  | `MPE_simple_crypto_v3`  |
| Simple Speaker Listener  | `MPE_simple_speaker_listener_v4`  |
| Simple Tag  | `MPE_simple_tag_v3`  |
| Simple World Comm | `MPE_simple_world_comm_v3` |
| Simple Reference | `MPE_simple_reference_v3` |
| Simple Adversary | `MPE_simple_adversary_v3` |


The implementations follow the PettingZoo code as closely as possible, including sharing variable names and version numbers. There are occasional discrepancies between the PettingZoo code and docs, where this occurs we have followed the code. As our implementation closely follows the PettingZoo code, please refer to their documentation for further information on the environments.

We additionally include a fully cooperative variant of Simple Tag, first used to evaluate FACMAC. In this environmnet, a number of agents attempt to tag a number of prey, where the prey are controlled by a heuristic AI.

| Envrionment  | JaxMARL Registry  |
|---|---|
| 3 agents, 1 prey  | `MPE_simple_facmac_3a_v1` |
| 6 agents, 2 prey  | `MPE_simple_facmac_6a_v1` |
| 9 agents, 3 prey  | `MPE_simple_facmac_9a_v1` |

## Action Space
Following the PettingZoo implementation, we allow for both discrete or continuous action spaces in all MPE envrionments. The environments use discrete actions by default.

**Discrete (default)**

Represents the combination of movement and communication actions. Agents that can move select a value 0-4 corresponding to `[do nothing, down, up, left, right]`, while agents that can communicate choose between a number of communication options. The agents' abilities and with the number of communication options varies with the envrionments.

**Continuous (`action_type="Continuous"`)**

Agnets that can move choose continuous values over `[do nothing, up, down, right, left]` with actions summed along their axis (i.e. vertical force = up - down). Agents that can communicate output values over the dimension of their communcation space. These two vectors are concatenated for agents that can move and communicate.

## Observation Space
The exact observation varies for each environment, but in general it is a vector of agent/landmark positions and velocities along with any communication values.

## Visualisation
Check the example `mpe_introduction.py` file in the tutorials folder for an introduction to our implementation of the MPE environments, including an example visualisation. We animate the environment after the state transitions have been collected as follows:

```python
import jax 
from jaxmarl import make
from jaxmarl.environments.mpe import MPEVisualizer

# Parameters + random keys
max_steps = 25
key = jax.random.PRNGKey(0)
key, key_r, key_a = jax.random.split(key, 3)

# Instantiate environment
env = make("MPE_simple_v3")
obs, state = env.reset(key_r)

# Sample random actions
key_a = jax.random.split(key_a, env.num_agents)
actions = {agent: env.action_space(agent).sample(key_a[i]) for i, agent in enumerate(env.agents)}

state_seq = []
for _ in range(max_steps):
    state_seq.append(state)
    # Iterate random keys and sample actions
    key, key_s, key_a = jax.random.split(key, 3)
    key_a = jax.random.split(key_a, env.num_agents)
    actions = {agent: env.action_space(agent).sample(key_a[i]) for i, agent in enumerate(env.agents)}

    # Step environment
    obs, state, rewards, dones, infos = env.step(key_s, state, actions)

# state_seq is a list of the jax env states passed to the step function
# i.e. [state_t0, state_t1, ...]
viz = MPEVisualizer(env, state_seq)
viz.animate(view=True)  # can also save the animiation as a .gif file with save_fname="mpe.gif"
```

## Citations
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

