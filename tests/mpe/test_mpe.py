"""
Test correspondance between pettingzoo and JaxMARL MPE environments
"""
import jax
import jax.numpy as jnp
import numpy as np
from pettingzoo.test import parallel_api_test
from pettingzoo.mpe import simple_v3, simple_world_comm_v3, simple_tag_v3, simple_spread_v3, simple_crypto_v3, simple_speaker_listener_v4, simple_push_v3, simple_adversary_v3, simple_reference_v3
import tqdm
from jaxmarl import make
import pytest

from jaxmarl.environments.mpe.default_params import DISCRETE_ACT, CONTINUOUS_ACT

num_episodes, num_steps, tolerance = 500, 25, 1e-4


def np_state_to_jax(env_zoo, env_jax):
    from jaxmarl.environments.mpe.simple import State

    p_pos = np.zeros((env_jax.num_entities, env_jax.dim_p))
    p_vel = np.zeros((env_jax.num_entities, env_jax.dim_p))
    c = np.zeros((env_jax.num_entities, env_jax.dim_c))
    for agent in env_zoo.aec_env.env.world.agents:
        a_idx = env_jax.a_to_i[agent.name]
        p_pos[a_idx] = agent.state.p_pos
        p_vel[a_idx] = agent.state.p_vel
        c[a_idx] = agent.state.c


    for landmark in env_zoo.aec_env.env.world.landmarks:
        l_idx = env_jax.l_to_i[landmark.name]
        p_pos[l_idx] = landmark.state.p_pos
        
    state = {
        "p_pos": p_pos,
        "p_vel": p_vel,
        "c": c,
        "step": env_zoo.aec_env.env.steps,
        "done": np.full((env_jax.num_agents), False),
    }
    
    if env_zoo.metadata["name"] == 'simple_crypto_v3':
        from jaxmarl.environments.mpe.simple_crypto import CryptoState
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

def assert_same_trans(step, obs_zoo, rew_zoo, done_zoo, obs_jax, rew_jax, done_jax, atol=1e-4):

    for agent in obs_zoo.keys():
        #print(f'{agent}: obs zoo {obs_zoo[agent]} len {len(obs_zoo[agent])}, obs jax {obs_jax[agent]} len {len(obs_jax[agent])}')
        assert np.allclose(obs_zoo[agent], obs_jax[agent], atol=atol), f"Step: {step}, observations for agent {agent} do not match. \nzoo obs: {obs_zoo}, \njax obs: {obs_jax}"
        assert np.allclose(rew_zoo[agent], rew_jax[agent], atol=atol), f"Step: {step}, Reward values for agent {agent} do not match, zoo rew: {rew_zoo[agent]}, jax rew: {rew_jax[agent]}"
        #print('done zoo', done_zoo, 'done jax', done_jax)
        assert np.alltrue(done_zoo[agent] == done_jax[agent]), f"Step: {step}, Done values do not match for agent {agent},  zoo done: {done_zoo[agent]}, jax done: {done_jax[agent]}"

def assert_same_state(env_zoo, env_jax, state_jax, atol=1e-4):

    state_zoo = np_state_to_jax(env_zoo, env_jax)
    
    for k in state_zoo.keys():
        jax_value = getattr(state_jax, k)
        if k not in ["step"]:        
            assert np.allclose(jax_value, state_zoo[k], atol=atol), f"State values do not match for key {k}, zoo value: {state_zoo[k]}, jax value: {jax_value}"

@pytest.mark.parametrize(("zoo_env_name", "action_type"),
                         [("MPE_simple_v3", DISCRETE_ACT),
                          ("MPE_simple_v3", CONTINUOUS_ACT),
                          ("MPE_simple_crypto_v3", DISCRETE_ACT),
                          ("MPE_simple_crypto_v3", CONTINUOUS_ACT),
                          ("MPE_simple_reference_v3", DISCRETE_ACT),
                          ("MPE_simple_reference_v3", CONTINUOUS_ACT),
                          ("MPE_simple_speaker_listener_v4", DISCRETE_ACT),
                          ("MPE_simple_speaker_listener_v4", CONTINUOUS_ACT),
                          ("MPE_simple_world_comm_v3", DISCRETE_ACT),
                          ("MPE_simple_world_comm_v3", CONTINUOUS_ACT),
                          ("MPE_simple_adversary_v3", DISCRETE_ACT),
                          ("MPE_simple_adversary_v3", CONTINUOUS_ACT),
                          ("MPE_simple_tag_v3", DISCRETE_ACT),
                          ("MPE_simple_tag_v3", CONTINUOUS_ACT),
                          ("MPE_simple_push_v3", DISCRETE_ACT),
                          ("MPE_simple_push_v3", CONTINUOUS_ACT),
                          ("MPE_simple_spread_v3", DISCRETE_ACT),
                          ("MPE_simple_spread_v3", CONTINUOUS_ACT),])
def test_mpe_vs_pettingzoo(zoo_env_name, action_type):
    print(f'-- Testing {zoo_env_name} --')
    key = jax.random.PRNGKey(0)
    
    if action_type == CONTINUOUS_ACT:
        continuous_actions=True
    else:
        continuous_actions=False
    
    env_zoo = zoo_mpe_env_mapper[zoo_env_name]

    env_zoo = env_zoo.parallel_env(max_cycles=25, continuous_actions=continuous_actions)
    zoo_obs = env_zoo.reset()

    env_jax = make(zoo_env_name, action_type=action_type)
    
    key, key_reset = jax.random.split(key)
    env_jax.reset(key_reset)
    for ep in tqdm.tqdm(range(num_episodes), desc=f"Testing {zoo_env_name}, epsiode:", leave=True):
        obs = env_zoo.reset()
        for s in range(num_steps):
            actions = {agent: env_zoo.action_space(agent).sample() for agent in env_zoo.agents}
            state = np_state_to_jax(env_zoo, env_jax)
            
            obs_zoo, rew_zoo, done_zoo, _, _ = env_zoo.step(actions)
            key, key_step = jax.random.split(key)
            obs_jax, state_jax, rew_jax, done_jax, _ = env_jax.step(key_step, state, actions)
            
            assert_same_trans(s, obs_zoo, rew_zoo, done_zoo, obs_jax, rew_jax, done_jax)
            
            if not np.alltrue(done_zoo.values()):
                assert_same_state(env_zoo, env_jax, state_jax)

    print(f'-- {zoo_env_name} all tests passed --')

zoo_mpe_env_mapper = {
    "MPE_simple_v3": simple_v3,
    "MPE_simple_world_comm_v3": simple_world_comm_v3,
    "MPE_simple_tag_v3": simple_tag_v3,
    "MPE_simple_spread_v3": simple_spread_v3,
    "MPE_simple_crypto_v3": simple_crypto_v3,
    "MPE_simple_speaker_listener_v4": simple_speaker_listener_v4,
    "MPE_simple_push_v3": simple_push_v3,
    "MPE_simple_adversary_v3": simple_adversary_v3,
    "MPE_simple_reference_v3": simple_reference_v3,
}

if __name__=="__main__":
    print(' *** Testing MPE ***')
    act_type = DISCRETE_ACT
    test_mpe_vs_pettingzoo("MPE_simple_v3", act_type)
    test_mpe_vs_pettingzoo("MPE_simple_crypto_v3", act_type)
    test_mpe_vs_pettingzoo("MPE_simple_reference_v3", act_type)
    test_mpe_vs_pettingzoo("MPE_simple_speaker_listener_v4", act_type)
    test_mpe_vs_pettingzoo("MPE_simple_world_comm_v3", act_type)
    test_mpe_vs_pettingzoo("MPE_simple_adversary_v3", act_type)
    test_mpe_vs_pettingzoo("MPE_simple_tag_v3", act_type)
    test_mpe_vs_pettingzoo("MPE_simple_push_v3", act_type)
    test_mpe_vs_pettingzoo("MPE_simple_spread_v3", act_type)    


    print(' *** All tests passed ***')
    
    
    
    
    
    