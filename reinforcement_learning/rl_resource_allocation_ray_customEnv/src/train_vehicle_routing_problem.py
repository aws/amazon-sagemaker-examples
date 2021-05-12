from model import register_actor_mask_model
from ray.tune.registry import register_env
from sagemaker_rl.ray_launcher import SageMakerRayLauncher

register_actor_mask_model()


class MyLauncher(SageMakerRayLauncher):
    def register_env_creator(self):
        from vrp_env import VRPGymEnvironment

        register_env("VRPGymEnvironment-v1", lambda env_config: VRPGymEnvironment(env_config))

    def get_experiment_config(self):
        return {
            "training": {
                "env": "VRPGymEnvironment-v1",
                "run": "APEX",
                "config": {
                    "double_q": False,
                    "dueling": False,
                    "num_atoms": 1,
                    "noisy": False,
                    "n_step": 3,
                    "lr": 0.0001,
                    "adam_epsilon": 0.00015,
                    "hiddens": [],
                    "buffer_size": 1000000,
                    "schedule_max_timesteps": 2000000,
                    "exploration_final_eps": 0.01,
                    "exploration_fraction": 0.1,
                    "prioritized_replay_alpha": 0.5,
                    "beta_annealing_fraction": 1.0,
                    "final_prioritized_replay_beta": 1.0,
                    "model": {
                        "custom_model": "action_mask",
                        "fcnet_hiddens": [512, 512],
                    },
                    # APEX
                    "num_workers": (self.num_cpus - 1),
                    "num_gpus": self.num_gpus,
                    "num_envs_per_worker": 8,
                    "sample_batch_size": 20,
                    "train_batch_size": 512,
                    "target_network_update_freq": 50000,
                    "timesteps_per_iteration": 25000,
                },
            }
        }


if __name__ == "__main__":
    MyLauncher().train_main()
