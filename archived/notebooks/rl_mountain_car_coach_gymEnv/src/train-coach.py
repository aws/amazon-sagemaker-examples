from sagemaker_rl.coach_launcher import SageMakerCoachPresetLauncher


class MyLauncher(SageMakerCoachPresetLauncher):
    def default_preset_name(self):
        """This points to a .py file that configures everything about the RL job.
        It can be overridden at runtime by specifying the RLCOACH_PRESET hyperparameter.
        """
        return "preset-mountaincarcontinuous-clippedppo.py"

    def map_hyperparameter(self, name, value):
        """Here we configure some shortcut names for hyperparameters that we expect to use frequently.
        Essentially anything in the preset file can be overridden through a hyperparameter with a name
        like "rl.agent_params.algorithm.etc".
        """
        # maps from alias (key) to fully qualified coach parameter (value)
        mapping = {
            "evaluation_freq_env_steps": "rl.steps_between_evaluation_periods:EnvironmentSteps",
            "evaluation_episodes": "rl.evaluation_steps:EnvironmentEpisodes",
            "improve_steps": "rl.improve_steps:EnvironmentSteps",
            "discount": "rl.agent_params.algorithm.discount",
            "gae_lambda": "rl.agent_params.algorithm.gae_lambda",
            "training_freq_env_steps": "rl.agent_params.algorithm.num_consecutive_playing_steps:EnvironmentSteps",
            "training_learning_rate": "rl.agent_params.network_wrappers.main.learning_rate",
            "training_batch_size": "rl.agent_params.network_wrappers.main.batch_size",
            "training_epochs": "rl.agent_params.algorithm.optimization_epochs",
        }
        if name in mapping:
            self.apply_hyperparameter(mapping[name], value)
        else:
            super().map_hyperparameter(name, value)

    def get_config_args(self, parser):
        args = super().get_config_args(parser)

        # Above line creates `self.hyperparameters` which is a collection of hyperparameters
        # that are to be added to graph manager when it's created. At this stage they are
        # fully qualified names since already passed through `map_hyperparameter`.

        # Keeps target network the same for all epochs of a single 'policy training' stage.
        src_hp = "rl.agent_params.algorithm.num_consecutive_playing_steps:EnvironmentSteps"
        target_hp = "rl.agent_params.algorithm.num_steps_between_copying_online_weights_to_target:EnvironmentSteps"
        if self.hyperparameters.hp_dict.get(src_hp, False):
            src_val = int(self.hyperparameters.hp_dict[src_hp])
            self.hyperparameters.hp_dict[target_hp] = src_val

        # Evaluate after each 'policy training' stage
        src_hp = "rl.agent_params.algorithm.num_consecutive_playing_steps:EnvironmentSteps"
        target_hp = "rl.steps_between_evaluation_periods:EnvironmentSteps"
        if self.hyperparameters.hp_dict.get(src_hp, False):
            src_val = int(self.hyperparameters.hp_dict[src_hp])
            self.hyperparameters.hp_dict[target_hp] = src_val

        return args


if __name__ == "__main__":
    MyLauncher.train_main()
