import tensorflow as tf

from . import tensorflow_train


class TensorflowInterface:
    """This is a base class that contains all the hidden tensorflow methods"""

    ### Static methods that interface with tensorflow.###
    @staticmethod
    def estimator_builder(name, model_params, model, remove_layers=None, ckpt=None):
        """Provides a tensorflow estimator out of a `Module` model.
        Args:
            name: Some name for the model.
            model_params: Params for the model builder.
            model: Model class.
            remove_layers: Either `None` or a list of booleans.
            ckpt: A checkpoint to begin training with.
        """
        run_config = tf.estimator.RunConfig(save_checkpoints_secs=60 * 60 * 24)

        if ckpt is not None:
            warm_start_settings = tf.estimator.WarmStartSettings(
                ckpt, vars_to_warm_start="^(?!.*dense)"
            )
        else:
            warm_start_settings = None

        return tf.estimator.Estimator(
            model_fn=model.builder,
            model_dir=model_params["dir"],
            config=run_config,
            warm_start_from=warm_start_settings,
            params=model_params,
        )

    @staticmethod
    def train(estimator, dataset, batch_size=128, epochs=1, epochs_between_evals=1):
        """A Method that trains a tensorflow model
        Args:
            estimator: An tf.Estimator object.
            dataset: A model of the dataset.Dataset class.
            batch_size: Mini-batch size to train.
            epochs: Number of epochs to train.
            epochs_between_evals: Frequency of evaluation on
                validation set.
        Returns:
            A dictionary of metrics.
        """
        retval = tensorflow_train(
            estimator=estimator,
            data_dir=dataset.data_dir,
            batch_size=batch_size,
            input_function=dataset.input_fn,
            epochs=epochs,
            epochs_between_evals=epochs_between_evals,
        )
        tf.summary.FileWriterCache.clear()
        return retval

    @staticmethod
    def _test():
        return None

    @staticmethod
    def get_dummy_symbol(dataset):
        """This methods returns a dummy input symbol. This can be used to create a graph just
        for reference purposes"""
        return tf.placeholder(
            tf.float32,
            shape=[None, dataset.height, dataset.width, dataset.num_channels],
            name="dummy_inputs",
        )
