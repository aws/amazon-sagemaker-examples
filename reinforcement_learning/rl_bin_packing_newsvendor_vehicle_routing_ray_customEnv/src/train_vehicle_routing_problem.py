from ray.tune.registry import register_env

from model import register_actor_mask_model
from sagemaker_rl.ray_launcher import SageMakerRayLauncher

register_actor_mask_model()


class MyLauncher(SageMakerRayLauncher):

    def register_env_creator(self):
        from vrp_environment import VRPGymEnvironment
        register_env("VRPGymEnvironment-v1", lambda env_config: VRPGymEnvironment(env_config))

    def get_experiment_config(self):
        return {
            "training": {
                "env": "VRPGymEnvironment-v1",
                "run": "PPO",
                "config": {
                    "gamma": 0.995,
                    "lambda": 0.95,
                    "clip_param": 0.2,
                    "kl_coeff": 1.0,
                    "num_sgd_iter": 10,
                    "lr": 0.0001,
                    "sample_batch_size": 1000,
                    "sgd_minibatch_size": 1024,
                    "train_batch_size": 35000,
                    "use_gae": False,
                    "num_workers": (self.num_cpus - 1),
                    "num_gpus": self.num_gpus,
                    "batch_mode": "complete_episodes",
                    "observation_filter": "NoFilter",
                    "model": {
                        "custom_model": "action_mask",
                        "fcnet_hiddens": [512, 512],
                    },
                    "vf_share_layers": False,
                    "entropy_coeff": 0.01,
                }
            }
        }


if __name__ == "__main__":
    MyLauncher().train_main()
