import io
import json
import logging
import os

import mxnet as mx

# Please make sure to import neomx
import neomx  # noqa: F401
import numpy as np

# Change the context to mx.gpu() if deploying to a GPU endpoint
ctx = mx.cpu()


def model_fn(model_dir):
    logging.info("Invoking user-defined model_fn")
    # The compiled model artifacts are saved with the prefix 'compiled'
    sym, arg_params, aux_params = mx.model.load_checkpoint(os.path.join(model_dir, "compiled"), 0)
    mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    exe = mod.bind(
        for_training=False, data_shapes=[("data", (1, 3, 224, 224))], label_shapes=mod._label_shapes
    )
    mod.set_params(arg_params, aux_params, allow_missing=True)
    # Run warm-up inference on empty data during model load (required for GPU)
    data = mx.nd.empty((1, 3, 224, 224), ctx=ctx)
    mod.predict(data)
    return mod


def transform_fn(mod, data, input_content_type, output_content_type):
    logging.info("Invoking user-defined transform_fn")
    if output_content_type == "application/json":
        # pre-processing
        data = json.loads(data)
        mx_ndarray = mx.nd.array(data)
        resized = mx.image.imresize(mx_ndarray, 224, 224)
        transposed = resized.transpose((2, 0, 1))
        batchified = transposed.expand_dims(axis=0)
        processed_input = batchified.as_in_context(ctx)

        # prediction/inference
        prediction_result = mod.predict(processed_input)

        # post-processing
        prediction = prediction_result.asnumpy().tolist()
        prediction_json = json.dumps(prediction[0])
        return prediction_json, output_content_type
