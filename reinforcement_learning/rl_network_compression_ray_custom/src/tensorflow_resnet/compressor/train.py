import logging
import math

import tensorflow as tf


def tensorflow_train(
    estimator, data_dir, batch_size, input_function, epochs=None, epochs_between_evals=1
):
    """
    This method will train a tensorflow model.

    Args:
        estimator: `tf.estimator.Estimator` object.
        data_dir: Directory where data is stored.
        batch_size: Mini batch size to train with.
        input_function: A function that will return a `tf.data.FixedLengthRecordDataset`.
        epochs: Number of epochs to train, if None will run eval only.
        epoch_between_evals: frequency of validation.
    """

    def input_fn_train(num_epochs):
        return input_function(
            is_training=True,
            data_dir=data_dir,
            batch_size=batch_size,
            num_epochs=num_epochs,
            dtype=tf.float32,
        )

    def input_fn_eval():
        return input_function(
            is_training=False,
            data_dir=data_dir,
            batch_size=batch_size,
            num_epochs=1,
            dtype=tf.float32,
        )

    if epochs is None:
        schedule, n_loops = [0], 1
    else:
        n_loops = math.ceil(epochs / epochs_between_evals)
        schedule = [epochs_between_evals for _ in range(int(n_loops))]
        schedule[-1] = epochs - sum(schedule[:-1])

    eval_results = None
    for cycle_index, num_train_epochs in enumerate(schedule):
        logging.info("Starting cycle: %d/%d", cycle_index, int(n_loops))

        if num_train_epochs:
            estimator.train(input_fn=lambda: input_fn_train(num_train_epochs))

        logging.info("Starting to evaluate.")
        eval_results = estimator.evaluate(input_fn=input_fn_eval)
        logging.info(eval_results)

    return eval_results
