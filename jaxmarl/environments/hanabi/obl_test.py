"""
Download the models with git clone https://huggingface.co/mttga/obl_flax in the same folder of this script

You need also the decks file here: decks_test.json
"""


import os
import jax
import numpy as np
from jax import numpy as jnp
from jaxmarl import make
from jaxmarl.wrappers.baselines import LogWrapper
from safetensors.flax import save_file, load_file
from flax.traverse_util import flatten_dict, unflatten_dict
from obl.obl_flax import SimpleOBLAgent, TorchAlignedLSTM 
import json

def load_params(filename):
    flattened_dict = load_file(filename)
    return unflatten_dict(flattened_dict, sep=',')

def save_array_to_file(arr, file_name):
    with open(file_name, "w") as file:
        for i, value in enumerate(arr):
            file.write(f"{i},{int(value)}\n")


env = LogWrapper(make('hanabi'))
flax_obl = SimpleOBLAgent()

env = LogWrapper(make('hanabi'))

batchify = lambda x: jnp.stack([x[a] for a in env.agents])
unbatchify = lambda x: {a:x[i] for i,a in enumerate(env.agents)}

@jax.jit
def run_obl_test(rng, params, return_all=False):

    def _env_step(carry, _):
    
        rng, env_state, agent_carry, last_obs = carry 
    
        _obs = batchify(last_obs)
        _legal_moves = batchify(env.get_legal_moves(env_state.env_state)) 
    
        flax_input = {
          'priv_s':_obs, # batch*agents, ...
          'publ_s':_obs[..., 125:], # batch*agents, ...
          'h0': agent_carry['h0'], # num_layer, batch, dim
          'c0': agent_carry['c0'],
          'legal_move': _legal_moves,
        }
        new_agent_carry = flax_obl.greedy_act(params, flax_input)
        actions = new_agent_carry.pop('a')

        aidx = jnp.nonzero(env_state.env_state.cur_player_idx, size=1)[0][0]
        
        actions = {agent:actions[i] for i, agent in enumerate(env.agents)}
        
        rng, _rng = jax.random.split(rng)
        new_obs, new_env_state, reward, dones, infos = env.step(_rng, env_state, actions)

        return (rng, new_env_state, new_agent_carry, new_obs), (infos, aidx, last_obs, actions, reward, dones, agent_carry)
    
    #rng, _rng = jax.random.split(rng)
    init_obs, env_state = env.reset(rng)
    
    agent_carry = {
      'h0': jnp.zeros((2, 2, 512)), # num_layer, batch_size, dim
      'c0':jnp.zeros((2, 2, 512)),
    }
    
    _, (infos, current_player, obs, actions, rewards, dones, agent_carries) = jax.lax.scan(
        _env_step, (rng, env_state, agent_carry, init_obs), None, 80
    )

    # compute the metrics of the first episode that is done for each parallel env
    def first_episode_returns(rewards, dones):
        first_done = jax.lax.select(jnp.argmax(dones)==0., dones.size, jnp.argmax(dones))
        first_episode_mask = jnp.where(jnp.arange(dones.size) <= first_done, True, False)
        return jnp.where(first_episode_mask, rewards, 0.).sum()
        
    returns = first_episode_returns(rewards['__all__'], dones['__all__'])

    if return_all:
        return infos, current_player, obs, actions, agent_carries
    return returns

def main():

    # load the decks
    with open('decks_test.json') as f:
        decks_j = json.load(f)
    decks_test_rngs = jnp.array([jnp.array(np.array(deck['jax_rng'], dtype=np.uint32))for deck in decks_j])

    f = jax.jit(jax.vmap(run_obl_test, in_axes=[0,None]))

    model_dir = 'obl/models/flax_models'
    output_dir = 'obl/deck_test_scores/flax_deck_scores'
    os.makedirs(output_dir, exist_ok=True)
    for d in sorted(os.listdir(model_dir)):
        print(d)
        for file in sorted(os.listdir(os.path.join(model_dir, d))):
            if file.split('.')[-1] == 'safetensors':
                params = load_params(os.path.join(model_dir, d, file))
                returns = f(decks_test_rngs, params)
                print('  ', file.split('.')[0], returns.mean())
                os.makedirs(os.path.join(output_dir, d), exist_ok=True)
                out_f = os.path.join(output_dir, d, file.split('.')[0]+'.txt')
                save_array_to_file(returns, out_f)

if __name__ == '__main__':
    main()
