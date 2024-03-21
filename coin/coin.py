
import jax 
import jax.numpy as jnp

import hydra 
from omegaconf import OmegaConf
import flax.linen as nn
from typing import Sequence
import distrax
from flax.linen.initializers import constant, orthogonal
from jaxmarl.environments.smax import map_name_to_scenario, HeuristicEnemySMAX
import tqdm

from evosax import OpenES, ParameterReshaper
from jaxmarl_coin_evalutator import SMAXCoinFitness


class RewardNetwork(nn.Module):
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        print('reward x shape', x.shape)
        critic = nn.Dense(
            64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0), name="reward_dense0"
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0), name="reward_dense1"
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0), name="reward_dense2")(
            critic
        )

        return jnp.squeeze(critic, axis=-1)

@hydra.main(version_base=None, config_path="configs", config_name="coin")
def main(config):
    config = OmegaConf.to_container(config)
    
    rng = jax.random.PRNGKey(config["SEED"])

    scenario = map_name_to_scenario(config["MAP_NAME"])
    env = HeuristicEnemySMAX(scenario=scenario, **config["ENV_KWARGS"])

    pholder = env.observation_space(env.agents[0]).shape[0]+1+3  # TODO not hardcode

    network = RewardNetwork()
    
    print('network setup', jnp.zeros((1, pholder)).shape)
    rng, rng_network = jax.random.split(rng)
    params = network.init(
        rng_network, 
        x=jnp.zeros((1, pholder)),
    )

    param_reshaper = ParameterReshaper(params)

    strategy = OpenES(popsize=2,
                    num_dims=param_reshaper.total_params,
                    opt_name="adam",
                    lrate_init=0.1
    )
    strategy.default_params

    num_generations = 10
    print_every_k_gens = 1
    
    evaluator = SMAXCoinFitness(config, n_devices=1)
    evaluator.set_apply_fn(network.apply)
    
    # o = evaluator.rollout_ffw(rng, params)
    # print('output', o)
    
    rng, rng_init = jax.random.split(rng)
    state = strategy.initialize(rng_init)
    for gen in tqdm.tqdm(range(num_generations)):
        
        rng, rng_ask, rng_eval = jax.random.split(rng, 3)
        x, state = strategy.ask(rng_ask, state)
        reshaped_params = param_reshaper.reshape(x)
        fitness = evaluator.rollout(rng_eval, reshaped_params)
        print('fitness', fitness)
        raise


if __name__=="__main__":
    main()