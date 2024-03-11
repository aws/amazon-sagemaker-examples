import logging
import time
from abc import ABC

import packaging.version
import requests
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from ts.handler_utils.distributed.pt_pippy import get_pipeline_driver
from ts.torch_handler.distributed.base_pippy_handler import BasePippyHandler

logger = logging.getLogger(__name__)
logger.info("Transformers version %s", transformers.__version__)
if packaging.version.parse(torch.__version__) >= packaging.version.parse("2.0.0"):
    logger.info("PyTorch version is 2.0.0 or greater")
else:
    logger.info(
        "PyTorch version is less than 2.0.0, initializing with meta device needs PyTorch 2.0.0 and greater"
    )


class TransformersSeqClassifierHandler(BasePippyHandler, ABC):
    """
    Transformers handler class for sequence, token classification and question answering.
    """

    def __init__(self):
        super(TransformersSeqClassifierHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        """In this initialize function, the HF large model is loaded and
        partitioned into multiple stages each on one device using PiPPy.
        Args:
            ctx (context): It is a JSON Object containing information
            pertaining to the model artefacts parameters.
        """
        super().initialize(ctx)
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        self.device = self.local_rank

        model_path = ctx.model_yaml_config["handler"]["model_path"]
        seed = ctx.model_yaml_config["handler"]["manual_seed"]
        dtype_str = ctx.model_yaml_config["handler"]["dtype"]
        torch.manual_seed(seed)

        dtypes = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

        dtype = dtypes.get(dtype_str, torch.float32)
        if dtype != torch.float32 and dtype_str not in dtypes:
            logger.info(
                f"Unsupported data type {dtype_str}, "
                "please submit a PR to support it. Falling back to fp32 now."
            )

        skip_init_start = time.perf_counter()
        with torch.device("meta"):
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, use_cache=False, torch_dtype=dtype
            )
        skip_init_end = time.perf_counter()
        logger.info(
            f" init model time on meta device took {skip_init_end - skip_init_start} seconds"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, return_tensors="pt")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_length = ctx.model_yaml_config["handler"]["max_length"]
        self.max_new_tokens = ctx.model_yaml_config["handler"]["max_new_tokens"]

        logger.info("Instantiating model Pipeline")
        pippy_compile_time_start = time.perf_counter()
        self.model = get_pipeline_driver(self.model, self.world_size, ctx)
        pippy_compile_time_end = time.perf_counter()

        logger.info(
            f" pippy compile time took {pippy_compile_time_end- pippy_compile_time_start} seconds on rank {self.local_rank}"
        )

        logger.info("Transformer model from path %s loaded successfully", model_dir)

        self.initialized = True

    def preprocess(self, requests):
        """
        Basic text preprocessing, based on the user's choice of application mode.
        Args:
            requests (list): A list of dictionaries with a "data" or "body" field, each
                            containing the input text to be processed.
        Returns:
            tuple: A tuple with two tensors: the batch of input ids and the batch of
                attention masks.
        """
        input_texts = [data.get("data") or data.get("body") for data in requests]
        input_ids_batch = []
        for input_text in input_texts:
            input_ids = self.encode_input_text(input_text)
            input_ids_batch.append(input_ids)
        input_ids_batch = torch.cat(input_ids_batch, dim=0).to(self.device)
        return input_ids_batch

    def encode_input_text(self, input_text):
        """
        Encodes a single input text using the tokenizer.
        Args:
            input_text (str): The input text to be encoded.
        Returns:
            tuple: A tuple with two tensors: the encoded input ids and the attention mask.
        """
        if isinstance(input_text, (bytes, bytearray)):
            input_text = input_text.decode("utf-8")
        logger.info("Received text: '%s'", input_text)
        inputs = self.tokenizer.encode_plus(
            input_text,
            max_length=self.max_length,
            pad_to_max_length=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"]
        return input_ids

    def inference(self, input_batch):
        """
        Predicts the class (or classes) of the received text using the serialized transformers
        checkpoint.
        Args:
            input_batch (tuple): A tuple with two tensors: the batch of input ids and the batch
                                of attention masks, as returned by the preprocess function.
        Returns:
            list: A list of strings with the predicted values for each input text in the batch.
        """
        input_ids_batch = input_batch
        input_ids_batch = input_ids_batch.to(self.device)
        outputs = self.model.generate(
            input_ids_batch,
            max_length=self.max_new_tokens,
        )
        generated_text = self.tokenizer.batch_decode(
            outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        logger.info("Generated text: %s", generated_text)
        return generated_text

    def postprocess(self, inference_output):
        """Post Process Function converts the predicted response into Torchserve readable format.
        Args:
            inference_output (list): It contains the predicted response of the input text.
        Returns:
            (list): Returns a list of the Predictions and Explanations.
        """
        return inference_output
