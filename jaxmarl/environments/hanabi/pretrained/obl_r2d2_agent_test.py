import os
import jax
import numpy as np
from jax import numpy as jnp
from jaxmarl import make
from jaxmarl.wrappers.baselines import load_params, LogWrapper
from jaxmarl.environments.hanabi.pretrained.obl_r2d2_agent import OBLAgentR2D2
import json

seed = 0
n_test_games = 1000
time_limit = 80
env = LogWrapper(make('hanabi'))
agent = OBLAgentR2D2()

batchify = lambda x: jnp.stack([x[a] for a in env.agents])
unbatchify = lambda x: {a:x[i] for i,a in enumerate(env.agents)}

@jax.jit
def run_obl_test(rng, params):

    def _env_step(carry, _):
    
        rng, env_state, agent_carry, last_obs = carry 

        agent_input = (
            batchify(last_obs),
            batchify(env.get_legal_moves(env_state.env_state))
        )

        new_agent_carry, actions = agent.greedy_act(params, agent_carry, agent_input)
        actions = unbatchify(actions)
        
        rng, _rng = jax.random.split(rng)
        new_obs, new_env_state, rewards, dones, infos = env.step(_rng, env_state, actions)

        return (rng, new_env_state, new_agent_carry, new_obs), (infos, rewards, dones)
    
    init_obs, env_state = env.reset(rng)
    agent_carry = agent.initialize_carry(jax.random.PRNGKey(0), batch_dims=(2,))

    _, (infos, rewards, dones) = jax.lax.scan(
        _env_step, (rng, env_state, agent_carry, init_obs), None, time_limit
    )

    # compute the metrics of the first episode that is done for each parallel env
    def first_episode_returns(rewards, dones):
        first_done = jax.lax.select(jnp.argmax(dones)==0., dones.size, jnp.argmax(dones))
        first_episode_mask = jnp.where(jnp.arange(dones.size) <= first_done, True, False)
        return jnp.where(first_episode_mask, rewards, 0.).sum()
        
    cum_rewards = first_episode_returns(rewards['__all__'], dones['__all__'])

    first_returned_episode = jnp.nonzero(infos['returned_episode'], size=1)[0][0]
    returns = infos['returned_episode_returns'][first_returned_episode][0]

    returns = jnp.where(
        first_returned_episode.any(), returns, cum_rewards
    )

    return returns

def main():
        
    rng = jax.random.PRNGKey(seed)
    test_rngs = jax.random.split(rng, n_test_games)
    f = jax.jit(jax.vmap(run_obl_test, in_axes=[0,None]))

    model_dir = './obl-r2d2-flax'
    for d in sorted(os.listdir(model_dir)):
        if not os.path.isdir(os.path.join(model_dir, d)):
            continue
        print(d)
        for file in sorted(os.listdir(os.path.join(model_dir, d))):
            if file.split('.')[-1] == 'safetensors':
                params = load_params(os.path.join(model_dir, d, file))
                returns = f(test_rngs, params)
                print('  ', file.split('.')[0], returns.mean())


if __name__ == '__main__':
    main()
