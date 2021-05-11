from model import register_actor_mask_model
from ray.tune.registry import register_env
from sagemaker_rl.ray_launcher import SageMakerRayLauncher

register_actor_mask_model()


class MyLauncher(SageMakerRayLauncher):
    def register_env_creator(self):
        from bin_packing_env import BinPackingActionMaskGymEnvironment

        register_env(
            "BinPackingActionMaskGymEnvironment-v1",
            lambda env_config: BinPackingActionMaskGymEnvironment(env_config),
        )

    def get_experiment_config(self):
        multi = 1
        return {
            "training": {
                "env": "BinPackingActionMaskGymEnvironment-v1",
                "run": "PPO",
                "config": {
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
                    "env_config": {
                        "bag_capacity": 9 * multi,
                        "item_sizes": [2 * multi, 3 * multi],
                        "item_probabilities": [0.75, 0.25],  # perfect pack -> SS: -20 to -100
                        # 'item_probabilities': [0.5, 0.5], #bounded waste ->  SS: -11 to -20
                        # 'item_probabilities': [0.8, 0.2], #linear waste -> SS: -150 to -340
                        "time_horizon": 10000,
                    },
                    "model": {
                        "custom_model": "action_mask",
                        "fcnet_hiddens": [256, 256],
                    },
                    "ignore_worker_failures": True,
                    "entropy_coeff": 0.01,
                },
                "checkpoint_freq": 1,  # make sure at least one checkpoint is saved
            }
        }


if __name__ == "__main__":
    MyLauncher().train_main()
