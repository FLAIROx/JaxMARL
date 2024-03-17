import jax
import jax.numpy as jnp
import chex
from flax.struct import dataclass
from typing import Tuple
from jaxmarl import make
from jaxmarl.gridworld.ma_maze import MAMaze
import math
from functools import partial
from jaxmarl.gridworld.grid_viz import GridVisualizer

@dataclass
class ArrayFixedSizeSet:
    """Fairly ridiculous class that can map from discrete arrays to unique integers.
    It needs linear time to both insert and retrieve elements, but does guarantee
    static sizing and can be passed to jit-ed functions etc. Uses a fixed placeholder
    value that must never occur.
    """

    data: chex.Array
    capacity: int
    array_shape: Tuple[int]
    size: int
    placeholder: int = 0

    @classmethod
    def create(cls, array_shape: Tuple[int], capacity: int, placeholder: int = 0):
        data = jnp.zeros((capacity, *array_shape)) + placeholder
        return cls(
            data=data,
            capacity=capacity,
            array_shape=array_shape,
            size=0,
            placeholder=placeholder,
        )

    def put(self, arr: chex.Array):
        idx = self.get(arr)
        data = jax.lax.cond(
            idx == -1, lambda: self.data.at[self.size].set(arr), lambda: self.data
        )
        size = jax.lax.cond(
            idx == -1,
            lambda: self.size + 1,
            lambda: self.size,
        )
        return self.replace(data=data, size=size)

    def get(self, arr: chex.Array):
        idx = jnp.argwhere(
            jnp.all(self.data == arr, axis=range(1, self.data.ndim)),
            size=1,
            fill_value=-1,
        )
        return idx.squeeze()


def main():

    epsilon = 0.2
    alpha = 0.1
    n_actions = 5
    gamma = 0.99
    env = MAMaze(height=3, width=3, n_walls=2)
    env_params = env.default_params
    buffer_capacity = 10000
    n_agents = 2
    rng = jax.random.PRNGKey(0)
    total_timesteps = 50000
    eval_timesteps = 1000

    def flatten_obs(obs):
        img = obs['image'].reshape((*obs['image'].shape[:-3], -1))
        return jnp.concatenate((img, jnp.expand_dims(obs['agent_dir'], 0)), axis=-1)

    def extract_obs(obs, idx):
        return jax.tree_map(lambda x: x[idx], obs)

    def sample_random_action(key):
        return jax.random.randint(key, shape=(), minval=0, maxval=n_actions)

    def sample_optimal_action(key, obs_storage, q_values, obs):
        q_index = obs_storage.get(flatten_obs(extract_obs(obs, 0)))
        max_action = jnp.argmax(q_values[q_index]).squeeze()
        random_action = sample_random_action(key)
        return jax.lax.cond(jnp.all(q_values[q_index] == q_values[q_index][0]), lambda: random_action, lambda: max_action)
    
    def sample_action(key, obs_storage, q_values, obs):

        key, key_rand_action, key_optimal_action = jax.random.split(key, num=3)
        optimal_action = sample_optimal_action(key=key_optimal_action, obs_storage=obs_storage, q_values=q_values, obs=obs)
        random_action = sample_random_action(key_rand_action)
        return jax.lax.cond(
            jax.random.uniform(key) < epsilon, lambda: random_action, lambda: optimal_action
        )

    @partial(jax.jit, static_argnums=(2,))
    def step(carry, _, evaluate=False):
        obs_storage, q_values, state, obs, cum_reward, key = carry
        key, action_key, opt_action_key = jax.random.split(key, num=3)
        action_key = jax.random.split(action_key, num=n_agents)
        if not evaluate:
            sampled_action = sample_action(opt_action_key, obs_storage, q_values, obs)
        else:
            sampled_action = sample_optimal_action(opt_action_key, obs_storage, q_values, obs)
        # actions = {
        #     agent: sample_random_action(action_key[i])
        #     for i, agent in enumerate(env.agents)
        # }
        # actions[env.agents[0]] = sampled_action
        actions = jnp.array([sampled_action, sample_random_action(action_key[1])])
        key, state_key = jax.random.split(key)
        next_obs, state, rewards, dones, infos = env.step(state_key, state, actions)
        cum_reward += rewards

        if not evaluate:
            obs_storage = obs_storage.put(flatten_obs(extract_obs(next_obs, 0)))
            q_index = obs_storage.get(flatten_obs(extract_obs(obs, 0)))
            next_q_index = obs_storage.get(flatten_obs(extract_obs(next_obs, 0)))
            updated_q_values = q_values.at[q_index, sampled_action].set(
                q_values[q_index, sampled_action]
                + alpha
                * (
                    gamma * jnp.max(q_values[next_q_index])
                    + rewards
                    - q_values[q_index, sampled_action]
                )
            )
            q_values = jax.lax.cond(dones, lambda: q_values, lambda: updated_q_values)
        return (obs_storage, q_values, state, obs, cum_reward, key), _


    obs_shape = math.prod(env.obs_shape) + 1
    obs_storage = ArrayFixedSizeSet.create((obs_shape,), capacity=buffer_capacity, placeholder=-1)
    q_values = jnp.zeros(shape=(buffer_capacity, n_actions))
    rng, rng_reset = jax.random.split(rng)
    obs, state = env.reset(rng_reset)
    init = (obs_storage, q_values, state, obs, 0, rng)
    (obs_storage, q_values, state, obs, cum_reward, rng), _ = jax.lax.scan(partial(step, evaluate=False), init=init, xs=None, length=total_timesteps)
    print(f"Total reward: {cum_reward}")
    rng, eval_rng_reset = jax.random.split(rng)
    obs, state = env.reset(eval_rng_reset)
    eval_init = (obs_storage, q_values, state, obs, 0, rng)
    (obs_storage, q_values, state, obs, cum_reward, rng), _ = jax.lax.scan(partial(step, evaluate=True), init=eval_init, xs=None, length=eval_timesteps)
    print(f"Total reward: {cum_reward}")
    viz = GridVisualizer()
    obs, state = env.reset(eval_rng_reset)
    viz.render(env_params, state, highlight=False)
    # TODO visualise the policy.
    for _ in range(1000):
        rng, action_key, opt_action_key = jax.random.split(rng, num=3)
        sampled_optimal_action = sample_optimal_action(opt_action_key, obs_storage, q_values, obs)
        actions = jnp.array([sampled_optimal_action, sample_random_action(action_key)])
        rng, state_key = jax.random.split(rng)
        obs, state, rewards, dones, infos = env.step(state_key, state, actions)
        viz.render(env_params, state, highlight=False)


if __name__ == "__main__":
    main()
