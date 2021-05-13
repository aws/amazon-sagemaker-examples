import knapsack_env
from sagemaker_rl.coach_launcher import SageMakerCoachPresetLauncher


class MyLauncher(SageMakerCoachPresetLauncher):
    def default_preset_name(self):
        """This points to a .py file that configures everything about the RL job.
        It can be overridden at runtime by specifying the RLCOACH_PRESET hyperparameter.
        """
        return "preset-knapsack-clippedppo"

    def map_hyperparameter(self, name, value):
        """Here we configure some shortcut names for hyperparameters that we expect to use frequently.
        Essentially anything in the preset file can be overridden through a hyperparameter with a name
        like "rl.agent_params.algorithm.etc".
        """
        if name == "warmup_latency":
            return self.apply_hyperparameter(
                "rl.env_params.additional_simulator_parameters.warmup_latency", value
            )
        if name == "discount":
            return self.apply_hyperparameter("rl.agent_params.algorithm.discount", value)
        if name == "online_to_target_steps":
            return self.apply_hyperparameter(
                "rl.agent_params.algorithm.num_steps_between_copying_online_weights_to_target:EnvironmentSteps",
                value,
            )
        if name == "eval_period":
            return self.apply_hyperparameter(
                "rl.steps_between_evaluation_periods:EnvironmentSteps", value
            )

        super().map_hyperparameter(name, value)


if __name__ == "__main__":
    MyLauncher.train_main()
