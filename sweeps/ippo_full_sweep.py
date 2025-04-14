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
        "min_iter": 5,  # 3 mins / Minimum iterations before early termination
        "eta": 2,
        "strict": True,
    },
    "parameters": {
       
        "NUM_STEPS": {
            "values": [32, 64, 128]  # default: 128
        },
        "NUM_ENVS": {
            "values": [8192, 16384, 32768, 65536]  
        },
        "NUM_MINIBATCHES": {
            "values": [ 256, 512, 1024, 2048]
        },
        "UPDATE_EPOCHS": {
            "min": 1,
            "max": 8,
            "distribution": "int_uniform" 
        },
        "ACTOR_ARCH": {
            "values": [
                [64, 64],
                [64, 64, 64], 
                [128, 64],   
                [128, 64, 64],          
            ]
        },
        "CRITIC_ARCH": {
            "values": [
              
                [128, 128, 128],      
                [128, 128, 128, 128],     
                [256, 128, 128],
            ]
        }
    }
}

# Initialize the sweep with wandb
sweep_id = wandb.sweep(sweep_config, project="single_quad_rl")
print("Sweep initialized with ID:", sweep_id)