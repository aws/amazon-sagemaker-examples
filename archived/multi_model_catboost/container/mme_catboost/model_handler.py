import os
import json
import sys
import logging
import time
import catboost
from catboost import CatBoostClassifier
import pandas as pd
import io

logger = logging.getLogger(__name__)

import os


class ModelHandler(object):
    def __init__(self):
        start = time.time()
        self.initialized = False
        print(f" perf __init__ {(time.time() - start) * 1000} ms")

    def initialize(self, ctx):
        start = time.time()
        self.device = "cpu"

        properties = ctx.system_properties
        self.device = "cpu"
        model_dir = properties.get("model_dir")

        print("model_dir {}".format(model_dir))
        print(os.system("ls {}".format(model_dir)))

        model_file = CatBoostClassifier()

        onlyfiles = [
            f
            for f in os.listdir(model_dir)
            if os.path.isfile(os.path.join(model_dir, f)) and f.endswith(".bin")
        ]
        print(
            f"Modelhandler:model_file location::{model_dir}:: files:bin:={onlyfiles} :: going to load the first one::"
        )
            
        self.model = model_file = model_file.load_model(onlyfiles[0])

        try:
            from os import getpid
            from os import getppid
            from threading import current_thread
            print(f'model_file={model_file}:: Process:Main pid={getpid()}, ppid={getppid()} thread={current_thread().name}')
        except:
            print("eror in get pid:ignore")

        self.initialized = True
        print(f" perf initialize {(time.time() - start) * 1000} ms")

    def preprocess(self, input_data):
        """
        Pre-process the request
        """

        start = time.time()
        print(type(input_data))
        output = input_data
        print(f" perf preprocess {(time.time() - start) * 1000} ms")
        return output

    def inference(self, inputs):
        """
        Make the inference request against the laoded model
        """
        start = time.time()

        predictions = self.model.predict_proba(inputs)
        print(f" perf inference {(time.time() - start) * 1000} ms")
        return predictions

    def postprocess(self, inference_output):
        """
        Post-process the request
        """

        start = time.time()
        inference_output = dict(enumerate(inference_output.flatten(), 0))
        print(f" perf postprocess {(time.time() - start) * 1000} ms")
        return [inference_output]

    def handle(self, data, context):
        """
        Call pre-process, inference and post-process functions
        :param data: input data
        :param context: mms context
        """
        start = time.time()

        input_data = data[0]["body"].decode()
        df = pd.read_csv(io.StringIO(input_data))

        model_input = self.preprocess(df)
        model_output = self.inference(model_input)
        print(f" perf handle in {(time.time() - start) * 1000} ms")
        return self.postprocess(model_output)


_service = ModelHandler()


def handle(data, context):
    start = time.time()
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    print(f" perf handle_out {(time.time() - start) * 1000} ms")
    return _service.handle(data, context)
