"""
Test transfer learning between the two environments. 

Initiliase from petting zoo, copy state into JAX initial state. Take actions 
with JAX policy. compare accumulated reward.
"""

import jax
import numpy as np
from jax import numpy as jnp
from smax import make
from baselines.QLearning.utils import load_params, get_space_dim
from baselines.QLearning.iql_ps import AgentRNN, ScannedRNN
from pettingzoo.mpe import simple_speaker_listener_v4


def np_state_to_jax(env_zoo, env_jax):
    from smax.environments.mpe.simple import State

    p_pos = np.zeros((env_jax.num_entities, env_jax.dim_p))
    p_vel = np.zeros((env_jax.num_entities, env_jax.dim_p))
    c = np.zeros((env_jax.num_entities, env_jax.dim_c))
    #print('--', env_zoo.aec_env.agents) # gives list of agent names
    #print('--', env_zoo.aec_env.env.world.agents)
    for agent in env_zoo.aec_env.env.world.agents:
        a_idx = env_jax.a_to_i[agent.name]
        #print('zoo agent pos', agent.state.p_pos)
        p_pos[a_idx] = agent.state.p_pos
        p_vel[a_idx] = agent.state.p_vel
        #print('Zoo communication state', agent.state.c)
        c[a_idx] = agent.state.c


    for landmark in env_zoo.aec_env.env.world.landmarks:
        l_idx = env_jax.l_to_i[landmark.name]
        #print('zoo landmark name', landmark.name)
        p_pos[l_idx] = landmark.state.p_pos
        #print('zoo landmark pos', landmark.state.p_pos)
        
    state = {
        "p_pos": p_pos,
        "p_vel": p_vel,
        "c": c,
        "step": env_zoo.aec_env.env.steps,
        "done": np.full((env_jax.num_agents), False),
    }
    
    #print('jax state', state)
    #print('test obs', state["p_pos"][1] - state["p_pos"][0])
    if env_zoo.metadata["name"] == 'simple_crypto_v3':
        from smax.environments.mpe.simple_crypto import CryptoState
        state["goal_colour"] = env_zoo.aec_env.env.world.agents[1].color
        state["private_key"] = env_zoo.aec_env.env.world.agents[2].key
        return CryptoState(**state)
    if env_zoo.metadata["name"] == 'simple_speaker_listener_v4':
        state["goal"] = int(env_zoo.aec_env.env.world.agents[0].goal_b.name[-1])
        return State(**state)
    if env_zoo.metadata["name"] == 'simple_push_v3' or env_zoo.metadata["name"] == 'simple_adversary_v3':
        state["goal"] = int(env_zoo.aec_env.env.world.agents[0].goal_a.name[-1])
        return State(**state)
    if env_zoo.metadata["name"] == 'simple_reference_v3':
        state["goal"] = np.flip(np.array([int(env_zoo.aec_env.env.world.agents[i].goal_b.name[-1]) for i in range(2)]))
        return State(**state)
    else:
        return State(**state)

key = jax.random.PRNGKey(42)
key, key_r = jax.random.split(key)

env_name = "MPE_simple_speaker_listener_v4"
env_jax = make(env_name)
env_jax.reset(key_r)

env_zoo = simple_speaker_listener_v4.parallel_env()
obs_zoo = env_zoo.reset()

state = np_state_to_jax(env_zoo, env_jax)

obs_jax = env_jax.get_obs(state)
print(obs_jax)
print(obs_zoo)




def _preprocess_obs(arr, extra_features):
    # flatten
    arr = arr.flatten()
    # pad the observation vectors to the maximum length
    pad_width = [(0, 0)] * (arr.ndim - 1) + [(0, max(0, max_obs_length - arr.shape[-1]))]
    arr = jnp.pad(arr, pad_width, mode='constant', constant_values=0)
    # concatenate the extra features
    arr = jnp.concatenate((arr, extra_features), axis=-1)
    return arr

pretrained_folder = "baselines/pretrained/"
alg_name = "iql_ps" # "vdn_ps", "qmix_ps"


# experiment parameters (necessary if you don't want to use the CT wrapper
max_obs_length = max(list(map(lambda x: get_space_dim(x), env_jax.observation_spaces.values())))
max_action_space = max(list(map(lambda x: get_space_dim(x), env_jax.action_spaces.values())))
valid_actions = {a:jnp.arange(get_space_dim(u)) for a, u in env_jax.action_spaces.items()}
agents_one_hot = {a:oh for a, oh in zip(env_jax.agents, jnp.eye(len(env_jax.agents)))}
agent_hidden_dim = 64

# agent network
agent = AgentRNN(max_action_space, agent_hidden_dim)
params = load_params(f'{pretrained_folder}/{env_name}/{alg_name}.safetensors')
if 'agent' in params.keys():
    params = params['agent'] # qmix also have mixer params

def obs_to_act(obs, dones):


    obs = jax.tree_util.tree_map(_preprocess_obs, obs, agents_one_hot)

    # add a dummy temporal dimension
    obs_   = jax.tree_map(lambda x: x[np.newaxis, np.newaxis, :], obs) # add also a dummy batch dim to obs
    dones_ = jax.tree_map(lambda x: x[np.newaxis, :], dones)

    # pass in one with homogeneous pass
    hstate = ScannedRNN.initialize_carry(agent_hidden_dim, len(env_jax.agents))
    hstate, q_vals = agent.homogeneous_pass(params, hstate, obs_, dones_)

    # get actions from q vals
    valid_q_vals = jax.tree_util.tree_map(lambda q, valid_idx: q.squeeze(0)[..., valid_idx], q_vals, valid_actions)
    actions = jax.tree_util.tree_map(lambda q: jnp.argmax(q, axis=-1).squeeze(0), valid_q_vals)
        
    return actions 

'''# reset

dones = {agent:jnp.zeros(1, dtype=bool) for agent in env.agents+['__all__']}
obs, state = env.reset(key)

# prepare inputs
obs = jax.tree_util.tree_map(_preprocess_obs, obs, agents_one_hot)

# add a dummy temporal dimension
obs_   = jax.tree_map(lambda x: x[np.newaxis, np.newaxis, :], obs) # add also a dummy batch dim to obs
dones_ = jax.tree_map(lambda x: x[np.newaxis, :], dones)

# pass in one with homogeneous pass
hstate = ScannedRNN.initialize_carry(agent_hidden_dim, len(env.agents))
hstate, q_vals = agent.homogeneous_pass(params, hstate, obs_, dones_)

# get actions from q vals
valid_q_vals = jax.tree_util.tree_map(lambda q, valid_idx: q.squeeze(0)[..., valid_idx], q_vals, valid_actions)
actions = jax.tree_util.tree_map(lambda q: jnp.argmax(q, axis=-1).squeeze(0), valid_q_vals)
'''

# zoo cycle
done_zoo = {agent:jnp.zeros(1, dtype=bool) for agent in env_jax.agents+['__all__']}

for _ in range(10):

    acts = obs_to_act(obs_zoo, done_zoo)
    obs_zoo, rew_zoo, done_zoo, _, _ = env_zoo.step(acts)

#obs, state, rewards, dones, info = env.step(key, state, actions)

#print(rewards)

