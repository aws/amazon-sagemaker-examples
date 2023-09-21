import logging
import os
from abc import ABC
from threading import Thread

import torch_neuronx
from transformers import AutoConfig, LlamaTokenizer
from transformers_neuronx.generation_utils import HuggingFaceGenerationModelAdapter
from transformers_neuronx.llama.model import LlamaForSampling

from ts.handler_utils.hf_batch_streamer import TextIteratorStreamerBatch
from ts.handler_utils.micro_batching import MicroBatching
from ts.protocol.otf_message_handler import send_intermediate_predict_response
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class LLMHandler(BaseHandler, ABC):
    """
    Transformers handler class for text completion streaming on Inferentia2
    """

    def __init__(self):
        super().__init__()
        self.initialized = False
        self.max_length = None
        self.tokenizer = None
        self.output_streamer = None
        # enable micro batching
        self.handle = MicroBatching(self)

    def initialize(self, ctx):
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        model_checkpoint_dir = ctx.model_yaml_config.get("handler", {}).get(
            "model_checkpoint_dir", ""
        )
        model_checkpoint_path = f"{model_dir}/{model_checkpoint_dir}"
        os.environ["NEURONX_CACHE"] = "on"
        os.environ["NEURONX_DUMP_TO"] = f"{model_dir}/neuron_cache"
        os.environ["NEURON_CC_FLAGS"] = "--model-type=transformer-inference"

        # micro batching initialization
        micro_batching_parallelism = ctx.model_yaml_config.get(
            "micro_batching", {}
        ).get("parallelism", None)
        if micro_batching_parallelism:
            logger.info(
                f"Setting micro batching parallelism  from model_config_yaml: {micro_batching_parallelism}"
            )
            self.handle.parallelism = micro_batching_parallelism

        micro_batch_size = ctx.model_yaml_config.get("micro_batching", {}).get(
            "micro_batch_size", 1
        )
        logger.info(f"Setting micro batching size: {micro_batch_size}")
        self.handle.micro_batch_size = micro_batch_size

        # settings for model compiliation and loading
        amp = ctx.model_yaml_config.get("handler", {}).get("amp", "f32")
        tp_degree = ctx.model_yaml_config.get("handler", {}).get("tp_degree", 6)
        self.max_length = ctx.model_yaml_config.get("handler", {}).get("max_length", 50)

        # allocate "tp_degree" number of neuron cores to the worker process
        os.environ["NEURON_RT_NUM_CORES"] = str(tp_degree)
        try:
            num_neuron_cores_available = (
                torch_neuronx.xla_impl.data_parallel.device_count()
            )
            assert num_neuron_cores_available >= int(tp_degree)
        except (RuntimeError, AssertionError) as error:
            logger.error(
                "Required number of neuron cores for tp_degree "
                + str(tp_degree)
                + " are not available: "
                + str(error)
            )

            raise error

        self.tokenizer = LlamaTokenizer.from_pretrained(model_checkpoint_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = LlamaForSampling.from_pretrained(
            model_checkpoint_path,
            batch_size=self.handle.micro_batch_size,
            amp=amp,
            tp_degree=tp_degree,
        )
        logger.info("Starting to compile the model")
        self.model.to_neuron()
        logger.info("Model has been successfully compiled")
        model_config = AutoConfig.from_pretrained(model_checkpoint_path)
        self.model = HuggingFaceGenerationModelAdapter(model_config, self.model)
        self.output_streamer = TextIteratorStreamerBatch(
            self.tokenizer,
            batch_size=self.handle.micro_batch_size,
            skip_special_tokens=True,
        )

        self.initialized = True

    def preprocess(self, requests):
        input_text = []
        for req in requests:
            data = req.get("data") or req.get("body")
            if isinstance(data, (bytes, bytearray)):
                data = data.decode("utf-8")
            logger.info(f"received req={data}")
            input_text.append(data.strip())

        # Ensure the compiled model can handle the input received
        if len(input_text) > self.handle.micro_batch_size:
            raise ValueError(
                f"Model is compiled for batch size {self.handle.micro_batch_size} but received input of size {len(input_text)}"
            )

        # Pad input to match compiled model batch size
        input_text.extend([""] * (self.handle.micro_batch_size - len(input_text)))

        return self.tokenizer(input_text, return_tensors="pt", padding=True)

    def inference(self, tokenized_input):
        generation_kwargs = dict(
            tokenized_input,
            streamer=self.output_streamer,
            max_new_tokens=self.max_length,
        )
        self.model.reset_generation()
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        micro_batch_idx = self.handle.get_micro_batch_idx()
        micro_batch_req_id_map = self.get_micro_batch_req_id_map(micro_batch_idx)
        for new_text in self.output_streamer:
            logger.debug("send response stream")
            send_intermediate_predict_response(
                new_text[: len(micro_batch_req_id_map)],
                micro_batch_req_id_map,
                "Intermediate Prediction success",
                200,
                self.context,
            )

        thread.join()

        return [""] * len(micro_batch_req_id_map)

    def postprocess(self, inference_output):
        return inference_output

    def get_micro_batch_req_id_map(self, micro_batch_idx: int):
        start_idx = micro_batch_idx * self.handle.micro_batch_size
        micro_batch_req_id_map = {
            index: self.context.request_ids[batch_index]
            for index, batch_index in enumerate(
                range(start_idx, start_idx + self.handle.micro_batch_size)
            )
            if batch_index in self.context.request_ids
        }

        return micro_batch_req_id_map
