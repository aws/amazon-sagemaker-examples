import shutil

from sagemaker_rl.coach_launcher import SageMakerCoachPresetLauncher


class MyLauncher(SageMakerCoachPresetLauncher):
    def default_preset_name(self):
        """This points to a .py file that configures everything about the RL job.
        It can be overridden at runtime by specifying the RLCOACH_PRESET hyperparameter.
        """
        return "preset-acrobot-dqn"

    def map_hyperparameter(self, name, value):
        """Here we configure some shortcut names for hyperparameters that we expect to use frequently.
        Essentially anything in the preset file can be overridden through a hyperparameter with a name
        like "rl.agent_params.algorithm.etc".
        """
        # maps from alias (key) to fully qualified coach parameter (value)
        mapping = {
            "discount": "rl.agent_params.algorithm.discount",
            "evaluation_episodes": "rl.evaluation_steps:EnvironmentEpisodes",
            "improve_steps": "rl.improve_steps:TrainingSteps",
        }
        if name in mapping:
            self.apply_hyperparameter(mapping[name], value)
        else:
            super().map_hyperparameter(name, value)

    def _save_tf_model(self):
        import tensorflow as tf

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

        # print([n.name for n in tf.get_default_graph().as_graph_def().node])
        # This is the node that will accept the input.
        input_nodes = tf.get_default_graph().get_tensor_by_name(
            "main_level/agent/main/online/" + "network_0/observation/observation:0"
        )
        # This is the node that will produce the output.
        output_nodes = tf.get_default_graph().get_operation_by_name(
            "main_level/agent/main/online/" + "network_0/q_values_head_0/softmax"
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
