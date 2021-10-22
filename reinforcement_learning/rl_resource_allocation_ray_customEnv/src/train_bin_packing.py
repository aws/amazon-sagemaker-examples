import sys
import json
# from model import register_actor_mask_model
from modelv2 import register_actor_mask_model
from ray.tune.registry import register_env
from sagemaker_rl.ray_launcher import SageMakerRayLauncher

from bin_packing_env import BinPackingActionMaskGymEnvironment

register_actor_mask_model()

config = {}


class MyLauncher(SageMakerRayLauncher):
    def register_env_creator(self):

        register_env(
            "BinPackingActionMaskGymEnvironment-v1",
            lambda env_config: BinPackingActionMaskGymEnvironment(env_config),
        )

    def get_experiment_config(self):
        print('~~~~~~~~~~~~~~~~~~~~~~~~~ get_experiment_config ~~~~~~~~~~~~~~~~~~~~~~~~')
        print(config)
        
        bag_capacity = int(config['bag_capacity'])
        item_sizes = json.loads(config['item_sizes'])
        item_probabilities = json.loads(config['item_probabilities'])
        time_horizon = int(config['time_horizon'])
                
        env_config = {
                        "bag_capacity": bag_capacity,
                        "item_sizes": item_sizes,
                        "item_probabilities": item_probabilities,  # perfect pack -> SS: -20 to -100
                        "time_horizon": time_horizon,
                    }
        
        env = BinPackingActionMaskGymEnvironment(env_config)
        obs_space_low = env.observation_space.spaces['real_obs'].low
        obs_space_high = env.observation_space.spaces['real_obs'].high
        
        return {
            "training": {
                "env": "BinPackingActionMaskGymEnvironment-v1",
                "run": "PPO",
                "config": {
                    "framework": "tf", 
                    "gamma": 0.995,
                    "kl_coeff": 1.0,
                    "num_sgd_iter": 10,
                    "lr": 0.0001,
                    "sgd_minibatch_size": 32768,
                    "train_batch_size": 320000,
                    "use_gae": False,
                    "num_workers": (self.num_cpus - 1),
                    "num_gpus": self.num_gpus,
                    "batch_mode": "complete_episodes",
                    "env_config": env_config,
                    "model": {
                        "custom_model": "action_mask",
                        "fcnet_hiddens": [256, 256],
                        "custom_model_config": {'obs_space_low': obs_space_low,
                                                'obs_space_high': obs_space_high},
                    },
                    "ignore_worker_failures": True,
                    "entropy_coeff": 0.01,
                },
                "checkpoint_freq": 1,  # make sure at least one checkpoint is saved
            }
        }


if __name__ == "__main__":
    for i in range(len(sys.argv)):
        if i == 0:
            continue
        if i % 2 > 0:
            config[sys.argv[i].split('--', 1)[1]] = sys.argv[i+1]
    MyLauncher().train_main()
