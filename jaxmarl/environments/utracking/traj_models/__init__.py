import json
import os

module_dir = os.path.dirname(__file__)
json_file_path = os.path.join(module_dir, 'traj_linear_models.json')

with open(json_file_path,'r') as f:
    traj_models = json.load(f)