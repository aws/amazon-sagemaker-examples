
import argparse

# import original framework mode script
import mnist

import tensorflow as tf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # read hyperparameters as script arguments
    parser.add_argument("--training_steps", type=int)
    parser.add_argument("--evaluation_steps", type=int)

    args, _ = parser.parse_known_args()

    # creates a tf.Estimator using `model_fn` that saves models to /opt/ml/model
    estimator = tf.estimator.Estimator(model_fn=mnist.model_fn, model_dir="/opt/ml/model")

    # creates parameterless input_fn function required by the estimator
    def input_fn():
        return mnist.train_input_fn(training_dir="/opt/ml/input/data/training", params=None)

    train_spec = tf.estimator.TrainSpec(input_fn, max_steps=args.training_steps)

    # creates parameterless serving_input_receiver_fn function required by the exporter
    def serving_input_receiver_fn():
        return mnist.serving_input_fn(params=None)

    exporter = tf.estimator.LatestExporter(
        "Servo", serving_input_receiver_fn=serving_input_receiver_fn
    )

    # creates parameterless input_fn function required by the evaluation
    def input_fn():
        return mnist.eval_input_fn(training_dir="/opt/ml/input/data/training", params=None)

    eval_spec = tf.estimator.EvalSpec(input_fn, steps=args.evaluation_steps, exporters=exporter)

    # start training and evaluation
    tf.estimator.train_and_evaluate(estimator=estimator, train_spec=train_spec, eval_spec=eval_spec)
