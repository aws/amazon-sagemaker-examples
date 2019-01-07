import logging
import os

"""
Writes Keras model to Tensorflow protobof. This is boiler plate code
"""


class ModelExporterKerasToProtobuf:

    @property
    def logger(self):
        return logging.getLogger(__name__)

    def __call__(self, path_to_h5, export_path):
        # Set the learning phase to Test since the model is already trained.
        # Import as local to avoid multiprocessing iusses
        from keras import backend as K
        from keras.engine.saving import load_model
        from tensorflow import Session, Graph
        from tensorflow.python.saved_model import builder as saved_model_builder
        from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def
        from tensorflow.saved_model import tag_constants
        K.set_learning_phase(0)

        with Graph().as_default():
            with Session().as_default() as sess:
                # Load the Keras model
                self.logger.info("Loading model {} to save as proto bof format at {}".format(path_to_h5, export_path))
                keras_model = load_model(path_to_h5)

                # Build the Protocol Buffer SavedModel at 'export_path'
                builder = saved_model_builder.SavedModelBuilder(export_path)

                # Create prediction signature to be used by TensorFlow Serving Predict API
                signature = predict_signature_def(inputs={"x": keras_model.input},
                                                  outputs={"y": keras_model.output})
                # Save the meta graph and the variables
                builder.add_meta_graph_and_variables(sess=sess, tags=[tag_constants.SERVING],
                                                     signature_def_map={"serving_default": signature})

                builder.save()

        saved_pb = os.path.join(export_path, "saved_model.pb")
        self.logger.info("Model saved to {}".format(saved_pb))
        return saved_pb
