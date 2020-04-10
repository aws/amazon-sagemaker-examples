"""
ModelHandler defines an example model handler for load and inference requests for MXNet CPU models
"""
from collections import namedtuple
import glob
import json
import logging
import os
import re

import mxnet as mx
import numpy as np

class ModelHandler(object):
    """
    A sample Model handler implementation.
    """

    def __init__(self):
        self.initialized = False
        self.mx_model = None
        self.shapes = None

    def get_model_files_prefix(self, model_dir):
        """
        Get the model prefix name for the model artifacts (symbol and parameter file).
        This assume model artifact directory contains a symbol file, parameter file, 
        model shapes file and a synset file defining the labels

        :param model_dir: Path to the directory with model artifacts
        :return: prefix string for model artifact files
        """
        sym_file_suffix = "-symbol.json"
        checkpoint_prefix_regex = "{}/*{}".format(model_dir, sym_file_suffix) # Ex output: /opt/ml/models/resnet-18/model/*-symbol.json
        checkpoint_prefix_filename = glob.glob(checkpoint_prefix_regex)[0] # Ex output: /opt/ml/models/resnet-18/model/resnet18-symbol.json
        checkpoint_prefix = os.path.basename(checkpoint_prefix_filename).split(sym_file_suffix)[0] # Ex output: resnet18
        logging.info("Prefix for the model artifacts: {}".format(checkpoint_prefix))
        return checkpoint_prefix

    def get_input_data_shapes(self, model_dir, checkpoint_prefix):
        """
        Get the model input data shapes and return the list

        :param model_dir: Path to the directory with model artifacts
        :param checkpoint_prefix: Model files prefix name
        :return: prefix string for model artifact files
        """
        shapes_file_path = os.path.join(model_dir, "{}-{}".format(checkpoint_prefix, "shapes.json"))
        if not os.path.isfile(shapes_file_path):
            raise RuntimeError("Missing {} file.".format(shapes_file_path))

        with open(shapes_file_path) as f:
            self.shapes = json.load(f)

        data_shapes = []

        for input_data in self.shapes:
            data_name = input_data["name"]
            data_shape = input_data["shape"]
            data_shapes.append((data_name, tuple(data_shape)))

        return data_shapes

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        self.initialized = True
        properties = context.system_properties
        # Contains the url parameter passed to the load request
        model_dir = properties.get("model_dir") 
        gpu_id = properties.get("gpu_id")

        checkpoint_prefix = self.get_model_files_prefix(model_dir)

        # Read the model input data shapes
        data_shapes = self.get_input_data_shapes(model_dir, checkpoint_prefix)
         
        # Load MXNet model
        try:
            ctx = mx.cpu() # Set the context on CPU
            sym, arg_params, aux_params = mx.model.load_checkpoint(checkpoint_prefix, 0)  # epoch set to 0
            self.mx_model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
            self.mx_model.bind(for_training=False, data_shapes=data_shapes, 
                   label_shapes=self.mx_model._label_shapes)
            self.mx_model.set_params(arg_params, aux_params, allow_missing=True)
            with open("synset.txt", 'r') as f:
                self.labels = [l.rstrip() for l in f]
        except (mx.base.MXNetError, RuntimeError) as memerr:
            if re.search('Failed to allocate (.*) Memory', str(memerr), re.IGNORECASE):
                logging.error("Memory allocation exception: {}".format(memerr))
                raise MemoryError
            raise           

    def preprocess(self, request):
        """
        Transform raw input into model input data.
        :param request: list of raw requests
        :return: list of preprocessed model input data
        """
        # Take the input data and pre-process it make it inference ready

        img_list = []
        for idx, data in enumerate(request):
            # Read the bytearray of the image from the input
            img_arr = data.get('body')  

            # Input image is in bytearray, convert it to MXNet NDArray
            img = mx.img.imdecode(img_arr) 
            if img is None:
                return None

            # convert into format (batch, RGB, width, height)
            img = mx.image.imresize(img, 224, 224) # resize
            img = img.transpose((2, 0, 1)) # Channel first
            img = img.expand_dims(axis=0) # batchify
            img_list.append(img)

        return img_list

    def inference(self, model_input):
        """
        Internal inference methods
        :param model_input: transformed model input data list
        :return: list of inference output in NDArray
        """
        # Do some inference call to engine here and return output
        Batch = namedtuple('Batch', ['data'])
        self.mx_model.forward(Batch(model_input))
        prob = self.mx_model.get_outputs()[0].asnumpy()
        return prob

    def postprocess(self, inference_output):
        """
        Return predict result in as list.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        # Take output from network and post-process to desired format
        prob = np.squeeze(inference_output)
        a = np.argsort(prob)[::-1]
        return [['probability=%f, class=%s' %(prob[i], self.labels[i]) for i in a[0:5]]]
        
    def handle(self, data, context):
        """
        Call preprocess, inference and post-process functions
        :param data: input data
        :param context: mms context
        """
        
        model_input = self.preprocess(data)
        model_out = self.inference(model_input)
        return self.postprocess(model_out)

_service = ModelHandler()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    return _service.handle(data, context)
