import wandb

# Define the sweep configuration as a nested dictionary
sweep_config = {
    "program": "train_multiquad_ippo.py",
    "method": "grid",
    "parameters": {
        "ENV_KWARGS.obs_noise": {
            "values": [1.0, 0.75, 0.5, 0.0]
        },
        "ENV_KWARGS.act_noise": {
            "values": [0.1, 0.05, 0.0]
        }
    }
}

# Initialize the sweep with wandb
sweep_id = wandb.sweep(sweep_config, project="single_quad_rl")
print("Sweep initialized with ID:", sweep_id)