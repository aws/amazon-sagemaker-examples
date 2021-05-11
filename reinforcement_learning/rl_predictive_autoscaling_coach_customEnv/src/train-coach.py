import shutil

import tensorflow as tf
from sagemaker_rl.coach_launcher import SageMakerCoachPresetLauncher


class MyLauncher(SageMakerCoachPresetLauncher):
    def default_preset_name(self):
        """This points to a .py file that configures everything about the RL job.
        It can be overridden at runtime by specifying the RLCOACH_PRESET hyperparameter.
        """
        return "preset-autoscale-ppo"

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

    def _save_tf_model(self):
        ckpt_dir = "/opt/ml/output/data/checkpoint"
        model_dir = "/opt/ml/model"

        # Re-Initialize from the checkpoint so that you will have the latest models up.
        tf.train.init_from_checkpoint(
            ckpt_dir, {"main_level/agent/online/network_0/": "main_level/agent/online/network_0"}
        )
        tf.train.init_from_checkpoint(
            ckpt_dir, {"main_level/agent/online/network_1/": "main_level/agent/online/network_1"}
        )

        # Create a new session with a new tf graph.
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        sess.run(tf.global_variables_initializer())  # initialize the checkpoint.

        # This is the node that will accept the input.
        input_nodes = tf.get_default_graph().get_tensor_by_name(
            "main_level/agent/main/online/" + "network_0/observation/observation:0"
        )
        # This is the node that will produce the output.
        output_nodes = tf.get_default_graph().get_operation_by_name(
            "main_level/agent/main/online/" + "network_1/ppo_head_0/policy_mean/BiasAdd"
        )
        # Save the model as a servable model.
        tf.saved_model.simple_save(
            session=sess,
            export_dir="model",
            inputs={"observation": input_nodes},
            outputs={"policy": output_nodes.outputs[0]},
        )
        # Move to the appropriate folder.
        shutil.move("model/", model_dir + "/model/tf-model/00000001/")
        # SageMaker will pick it up and upload to the right path.
        print("Success")


if __name__ == "__main__":
    MyLauncher.train_main()
