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
from jaxmarl import make
from jaxmarl.environments.smax import map_name_to_scenario
from jaxmarl.environments.smax.heuristic_enemy import (
    create_heuristic_policy,
    get_heuristic_policy_initial_state,
)
from jaxmarl.viz.visualizer import Visualizer, SMAXVisualizer
import os
from typing import Sequence

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
# Parameters + random keys
max_steps = 15
key = jax.random.PRNGKey(2)
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
    scenario = map_name_to_scenario("5m_vs_6m")
    env = make(
        "HeuristicEnemySMAX",
        # attack_mode="random",
        scenario=scenario,
        use_self_play_reward=False,
        walls_cause_death=True,
        see_enemy_actions=False,
        # num_allies=3,
        # num_enemies=5,
        # smacv2_position_generation=True,
        # smacv2_unit_type_generation=True,
        action_type="continuous",
        observation_type="conic"
    )
    # env = make("SMAX")
    # params = init_policy(env, key_p)
    # learned_policy = LearnedPolicy(env.action_space(env.agents[0]).n, activation="tanh")
    # env = make("LearnedPolicyEnemySMAX", params=params, policy=learned_policy)
    obs, state = env.reset(key_r)
    print("list of agents in environment", env.agents)

    # Sample random actions
    key_a = jax.random.split(key_a, env.num_agents)
    actions = {
        agent: env.action_space(agent).sample(key_a[i])
        for i, agent in enumerate(env.agents)
    }
    print("example action dict", actions)

    policy_states = {
        agent: get_heuristic_policy_initial_state() for agent in env.agents
    }
    policy = create_heuristic_policy(env, 0, shoot=True, attack_mode="closest")
    enemy_policy = create_heuristic_policy(env, 1, shoot=True, attack_mode="closest")
    state_seq = []
    returns = {a: 0 for a in env.agents}
    for i in range(max_steps):
        # Iterate random keys and sample actions
        key, key_s, key_seq = jax.random.split(key, 3)
        key_a = jax.random.split(key_seq, env.num_agents)
        actions = {}
        # for i, agent in enumerate(env.agents):
        #     p = policy if i < env.num_allies else enemy_policy
        #     action, policy_state = p(key_a[i], policy_states[agent], obs[agent])
        #     policy_states[agent] = policy_state
        #     actions[agent] = action

        # actions = {agent: jnp.array(1) for agent in env.agents}
        actions = {agent: env.action_space(agent).sample(key_a[i]) for i, agent in enumerate(env.agents)}
        state_seq.append((key_s, state, actions))
        # Step environment
        avail_actions = env.get_avail_actions(state)
        obs, state, rewards, dones, infos = env.step(key_s, state, actions)
        print(f"Actions: {actions}")
        returns = {a: returns[a] + rewards[a] for a in env.agents}
        if dones["__all__"]:
            print(f"Returns: {returns}")
print(f"Returns: {returns}")
viz = SMAXVisualizer(env, state_seq)

viz.animate(view=False, save_fname="output.gif")
