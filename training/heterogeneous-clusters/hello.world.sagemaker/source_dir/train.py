import json
import os
import sys
from sagemaker_training import environment # This module is present on the DLC images, or you can install it with pip install sagemaker_training

if __name__ == "__main__":
    
    print("Option-1: Read instance group information from the sagemaker_training.environment.Environment class")
    env = environment.Environment()    
    print(f"env.is_hetero: {env.is_hetero}")
    print(f"env.current_host: {env.current_host}")
    print(f"env.current_instance_type: {env.current_instance_type}")
    print(f"env.current_instance_group: {env.current_instance_group}")
    print(f"env.current_instance_group_hosts: {env.current_instance_group_hosts}")
    print(f"env.instance_groups: {env.instance_groups}")
    print(f"env.instance_groups_dict: {env.instance_groups_dict}")
    print(f"env.distribution_hosts: {env.distribution_hosts}")
    print(f"env.distribution_instance_groups: {env.distribution_instance_groups}")
        

    file_path = '/opt/ml/input/config/resourceconfig.json'
    print("Option-2: Read instance group information from {file_path}.\
            You'll need to parse the json yourself. This doesn't require an additional library.\n")
    
    with open(file_path, 'r') as f:
        config = json.load(f)

    print(f'{file_path} dump = {json.dumps(config, indent=4, sort_keys=True)}')
    
    print(f"env.is_hetero: {'instance_groups' in config}")
    print(f"current_host={config['current_host']}")
    print(f"current_instance_type={config['current_instance_type']}")
    print(f"env.current_instance_group: {config['current_group_name']}")
    print(f"env.current_instance_group_hosts: TODO")
    print(f"env.instance_groups: TODO")
    print(f"env.instance_groups_dict: {config['instance_groups']}")
    print(f"env.distribution_hosts: TODO")
    print(f"env.distribution_instance_groups: TODO")
