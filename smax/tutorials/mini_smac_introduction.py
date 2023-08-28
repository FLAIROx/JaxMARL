"""
Short introduction to the package.

## Abstract base class
Uses the PettingZoo Parallel API. All agents act synchronously, with actions, 
observations, returns and dones passed as dictionaries keyed by agent names. 
The code can be found in `multiagentgymnax/multi_agent_env.py`

The class follows an identical structure to that of `gymnax` with one execption. 
The class is instatiated with `num_agents`, defining the number of agents within the environment.

## Environment loop
Below is an example of a simple environment loop, using random actions.

"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import distrax
from smax import make
from smax.environments.mini_smac import map_name_to_scenario
from smax.environments.mini_smac.heuristic_enemy import create_heuristic_policy
from smax.viz.visualizer import Visualizer, MiniSMACVisualizer
import os
from typing import Sequence

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
# Parameters + random keys
max_steps = 30
key = jax.random.PRNGKey(1)
key, key_r, key_a, key_p = jax.random.split(key, 4)


class LearnedPolicy(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(self.action_dim)(x)
        actor_mean = activation(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(1)(x)
        critic = activation(critic)

        return pi, jnp.squeeze(critic, axis=-1)


def init_policy(env, rng):
    network = LearnedPolicy(env.action_space(env.agents[0]).n, activation="tanh")
    rng, _rng = jax.random.split(rng)
    init_x = jnp.zeros(env.observation_space(env.agents[0]).shape)
    params = network.init(_rng, init_x)
    return params


# Instantiate environment
with jax.disable_jit(False):
    scenario = map_name_to_scenario("3m")
    env = make(
        "HeuristicEnemyMiniSMAC",
        enemy_shoots=True,
        scenario=scenario,
        num_agents_per_team=3,
        use_self_play_reward=False,
        walls_cause_death=True,
        see_enemy_actions=False,
    )
    # env = make("MiniSMAC")
    # params = init_policy(env, key_p)
    # learned_policy = LearnedPolicy(env.action_space(env.agents[0]).n, activation="tanh")
    # env = make("LearnedPolicyEnemyMiniSMAC", params=params, policy=learned_policy)
    obs, state = env.reset(key_r)
    print("list of agents in environment", env.agents)

    # Sample random actions
    key_a = jax.random.split(key_a, env.num_agents)
    actions = {
        agent: env.action_space(agent).sample(key_a[i])
        for i, agent in enumerate(env.agents)
    }
    print("example action dict", actions)

    policy = create_heuristic_policy(env, 0, shoot=True)
    state_seq = []
    returns = {a: 0 for a in env.agents}
    for i in range(max_steps):
        # Iterate random keys and sample actions
        key, key_s, key_seq = jax.random.split(key, 3)
        key_a = jax.random.split(key_seq, env.num_agents)
        actions = {
            agent: policy(key_a[i], obs[agent]) for i, agent in enumerate(env.agents)
        }
        # actions = {agent: jnp.array(1) for agent in env.agents}
        # actions = {agent: env.action_space(agent).sample(key_a[i]) for i, agent in enumerate(env.agents)}
        state_seq.append((key_s, state, actions))
        # Step environment
        obs, state, rewards, dones, infos = env.step(key_s, state, actions)
        returns = {a: returns[a] + rewards[a] for a in env.agents}
        if dones["__all__"]:
            print(f"Returns: {returns}")

print(f"Returns: {returns}")
viz = MiniSMACVisualizer(env, state_seq)

viz.animate(view=False, save_fname="output.gif")
