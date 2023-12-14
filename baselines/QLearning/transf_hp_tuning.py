import os
import jax
import importlib.util

def hyperparam_search(
    rng,
    file_path:str,  # The path to the file to be parameterized
    config:dict,  # Configuration of the alg
    hyper_param_space:dict,  # Hyperparameter space dictionary  
    seeds_per_exp:int=2,  # The number of seeds per experiment, default is 2
    function_name:str='make_train',  # The name of the function to be parameterized, default is 'make_train'
    subfunction_name:str='train',  # The name of the subfunction to be parameterized, default is 'train'
    remove_tmp_file:bool=True,  # Whether to remove the temporary file after use, default is True
    env=None, # jaxmarl env
): 
    """
    Perform a hyperparameter search by creating a new parameterized file from a given file,
    importing the new modified function, and then training the model with the properly vmapped new function.
    """

    # WRITE A NEW PARAMETRIZED FILE
    replacements = {
        **{f'config["{param}"]':param for param in hyper_param_space.keys()},
        **{f"config['{param}']":param for param in hyper_param_space.keys()}
    }
    
    with open(file_path, 'r') as file:
        script = file.readlines()

    new_script = []
    in_function = False
    in_subfuction = False
    for line in script:
        stripped = line.strip()
        if stripped.startswith('def ' + function_name):
            in_function = True
        elif in_function and stripped.startswith('def '+ subfunction_name):
            # Add new parameters to the function definition
            in_subfuction = True
            line = line.replace('):', ', ' + ', '.join(hyper_param_space.keys()) + '):')
        elif stripped == 'return train':
            in_function = False

        if in_subfuction:
            for old, new in replacements.items():
                line = line.replace(old, new)

        new_script.append(line)

    # Create a new file with the 'tmp' suffix
    base, ext = os.path.splitext(file_path)
    new_file_path = base + '_tmp' + ext
    with open(new_file_path, 'w') as file:
        file.writelines(new_script)

    # Import the new, modified function
    spec = importlib.util.spec_from_file_location(function_name, new_file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    new_function = getattr(module, function_name)

    # Delete the temporary file
    if remove_tmp_file:
        os.remove(new_file_path)

    # VMAP THE TRAINING FUNCTION
    if env is not None:
        train_vmapped = new_function(config, env)
    else:
        train_vmapped = new_function(config)
    for i in range(len(hyper_param_space), -1, -1):
        vmap_map = [None]*(len(hyper_param_space)+1)
        vmap_map[i] = 0
        train_vmapped = jax.vmap(train_vmapped, in_axes=vmap_map)

    # TRAIN
    rngs = jax.random.split(rng, seeds_per_exp)
    outs = jax.jit(train_vmapped)(rngs, *hyper_param_space.values())
    
    return outs


def main():
    from jaxmarl import make
    from jaxmarl.wrappers.baselines import SMAXLogWrapper
    from jaxmarl.environments.smax import map_name_to_scenario
    from jax import numpy as jnp
    import itertools
    import wandb

    train_script = '/app/JaxMARL/baselines/QLearning/transf_qmix_mpe.py'

    env = make(
        "MPE_simple_spread_v3"
    )
    #env = SMAXLogWrapper(env)

    config = {
        "NUM_ENVS": 8,
        "NUM_STEPS":25,
        "BUFFER_SIZE": 3000,
        "BUFFER_BATCH_SIZE": 32,
        "TOTAL_TIMESTEPS": 10000,
        "AGENT_INIT_SCALE": 2.,
        "AGENT_HIDDEN_DIM": 32,
        "AGENT_TRANSF_NUM_LAYERS": 2,
        "AGENT_TRANSF_NUM_HEADS": 4,
        "AGENT_TRANSF_DIM_FF": 128,
        "PARAMETERS_SHARING": True,
        "EPSILON_START": 1.0,
        "EPSILON_FINISH": 0.05,
        "EPSILON_ANNEAL_TIME": 100000,
        "MIXER_INIT_SCALE": 0.0001,
        "MIXER_TRANSF_NUM_LAYERS": 2,
        "MIXER_TRANSF_NUM_HEADS": 4,
        "MIXER_TRANSF_DIM_FF": 128,
        "MAX_GRAD_NORM": 25,
        "TARGET_UPDATE_INTERVAL": 200,
        "LR": 0.0001,
        "LR_LINEAR_DECAY": False,
        "EPS_ADAM": 0.001,
        "WEIGHT_DECAY_ADAM": 0.00001,
        "TD_LAMBDA_LOSS": True,
        "TD_LAMBDA": 0.6,
        "GAMMA": 0.9,
        "VERBOSE": True,
        "WANDB_ONLINE_REPORT": False,
        "NUM_TEST_EPISODES": 32,
        "TEST_INTERVAL": 50000,
        "ENTITY": "mttga",
        "PROJECT": "jaxMARL_mpe",
        "WANDB_MODE" : "online",
    }

    # not vmapped params
    static_param_space =  {
        'AGENT_INIT_SCALE':[1, 0.0001],
        'MIXER_INIT_SCALE':[1, 0.0001],
        'AGENT_TRANSF_NUM_HEADS':[4, 8],
        'MIXER_TRANSF_NUM_HEADS':[4, 8],
        'LR_LINEAR_DECAY':[True,False],
    }

    # vmapped params
    hyper_param_space = {
        'LR':jnp.array([0.005, 0.0005]),
        'EPS_ADAM':jnp.array([0.00001, 0.0000001]),
    }

    # run the not-vmapped experiments
    for s_idx in itertools.product(*map(lambda x: list(range(len(x))), static_param_space.values())):

        # change the config file
        for i, k in zip(s_idx, static_param_space):
            config[k] = static_param_space[k][i]

        print(config)

        static_label = "_".join(f'{k}={static_param_space[k][i]:.5f}' for i, k in zip(s_idx, static_param_space))

        # run the vmapped experiments
        outs = hyperparam_search(
            file_path=train_script,
            config=config,
            hyper_param_space=hyper_param_space,
            env=env,
            rng=jax.random.PRNGKey(30),
            seeds_per_exp=1,
        )

        # log the results as separate metrics
        log_metrics = {
            'timesteps':outs['metrics']['timesteps'][0],
            'returns':outs['metrics']['rewards']['__all__'].mean(axis=0),
            'loss':outs['metrics']['loss'].mean(axis=0),
        }

        for idx in itertools.product(*map(lambda x: list(range(len(x))), hyper_param_space.values())):
            label = "_".join(f'{k}={hyper_param_space[k][i]:.5f}' for i, k in zip(idx, hyper_param_space))
            exp_name = f'transfqmix_{static_label}_{label}'

            run = wandb.init(
                entity=config["ENTITY"],
                project=config["PROJECT"],
                tags=["TRANSFORMER"],
                name=exp_name,
                config=config,
                mode=config["WANDB_MODE"],
                group='transf_qmix_mpe_ht',
            )

            run_logs = jax.tree_util.tree_map(lambda x: x[idx].tolist(), log_metrics)
            for values in zip(*run_logs.values()):
                run.log({k:v for k, v in zip(run_logs.keys(), values)})

            run.finish()


if __name__=='__main__':
    main()