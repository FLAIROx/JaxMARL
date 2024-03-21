
import jax 
import jax.numpy as jnp

from jaxmarl.environments.smax import map_name_to_scenario, HeuristicEnemySMAX

from evosax import OpenES, NetworkMapper, ParameterReshaper


config = {
    "MAP_NAME": "3m",
    "ENV_KWARGS": {},
}

rng = jax.random.PRNGKey(0)

scenario = map_name_to_scenario(config["MAP_NAME"])
env = HeuristicEnemySMAX(scenario=scenario, **config["ENV_KWARGS"])

pholder = env.observation_space(env.agents[0]).shape[0]+1

network = NetworkMapper["MLP"](
    num_hidden_units=64,
    num_hidden_layers=2,
    num_output_units=1,
    hidden_activation="relu",
)

rng, rng_network = jax.random.split(rng)
params = network.init(
    rng_network, 
    x=jnp.zeros((1, pholder))
)

param_reshaper = ParameterReshaper(params)

strategy = OpenES(popsize=100,
                num_dims=param_reshaper.total_params,
                opt_name="adam",
                lrate_init=0.1)
strategy.default_params


num_generations = 10
print_every_k_gens = 1

