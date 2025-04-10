import wandb

# Define the sweep configuration as a nested dictionary
sweep_config = {
    "program": "train_multiquad_ippo.py",
    "method": "bayes",
    "metric": {
        "name": "episode_length_interval",  # Metric to maximize
        "goal": "maximize"
    },
    "early_terminate": {
        "type": "hyperband",
        "min_iter": 3,  # 3 mins / Minimum iterations before early termination
        "eta": 2,
        "strict": True,
    },
    "parameters": {
        "LR": {
            "distribution": "log_uniform_values", 
            "min": 1e-4,
            "max": 1e-3  # default: 3e-4
        },
        "NUM_STEPS": {
            "values": [32, 64, 128, 256, 512, 1024, 2048]  # default: 128
        },
        "NUM_ENVS": {
            "values": [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]  # default: 2048
        },
        "NUM_MINIBATCHES": {
            "values": [2, 4, 8, 16, 32, 64, 128, 256, 512]  # default: 8
        },
        "UPDATE_EPOCHS": {
            "min": 1,
            "max": 32,
            "distribution": "int_uniform" 
        },
        "ACTOR_ARCH": {
            "values": [
                [64, 64],
                [64, 64, 64], 
                [64, 64, 64, 64],
                [128, 64, 64, 64],
                [128, 64, 64],         
                [128, 128, 128]   
            ]
        },
        "CRITIC_ARCH": {
            "values": [
                [64, 64],
                [64, 64, 64], 
                [64, 64, 64, 64],
                [128, 128],
                [128, 128, 64, 64],   
                [128, 128, 128],      
                [256, 128, 128], 
                [256, 256, 256],
                [128, 128, 128, 128],
                [256, 256, 128, 128],
                [256, 256, 256, 256],   
                [128, 128, 128, 128, 128],
            ]
        }
    }
}

# Initialize the sweep with wandb
sweep_id = wandb.sweep(sweep_config, project="single_quad_rl")
print("Sweep initialized with ID:", sweep_id)