from ray.tune.registry import register_env
from sagemaker_rl.ray_launcher import SageMakerRayLauncher


class MyLauncher(SageMakerRayLauncher):
    def register_env_creator(self):
        from news_vendor_env import NewsVendorGymEnvironmentNormalized

        register_env(
            "NewsVendorGymEnvironment-v1",
            lambda env_config: NewsVendorGymEnvironmentNormalized(env_config),
        )

    def get_experiment_config(self):
        return {
            "training": {
                "env": "NewsVendorGymEnvironment-v1",
                "run": "PPO",
                "config": {
                    "ignore_worker_failures": True,
                    "gamma": 1,
                    "kl_coeff": 1.0,
                    "num_sgd_iter": 5,
                    "lr": 0.0001,
                    "sgd_minibatch_size": 32768,
                    "train_batch_size": 320000,
                    "model": {
                        "fcnet_hiddens": [64, 64],
                    },
                    "use_gae": False,
                    "num_workers": (self.num_cpus - 1),
                    "num_gpus": self.num_gpus,
                    "batch_mode": "complete_episodes",
                    "env_config": {},
                    "observation_filter": "MeanStdFilter",
                },
                "checkpoint_freq": 2,
            }
        }


if __name__ == "__main__":
    MyLauncher().train_main()
