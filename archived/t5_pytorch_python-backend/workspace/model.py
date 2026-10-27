import numpy as np
import sys
import os
import json
from pathlib import Path

import torch

import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    # Every Python model must have "TritonPythonModel" as the class name!
    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.
        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        self.output_dtype = pb_utils.triton_string_to_numpy(
            pb_utils.get_output_config_by_name(
                json.loads(args['model_config']), "output"
            )['data_type']
        )

        from transformers import T5ForConditionalGeneration, T5Tokenizer
        self.model = T5ForConditionalGeneration.from_pretrained("t5-small").cuda()
        print("TritonPythonModel initialized")

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model.
        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """
        responses = []
        for request in requests:
            input_ids = pb_utils.get_input_tensor_by_name(request, "input_ids")
            input_ids = input_ids.as_numpy()
            input_ids = torch.as_tensor(input_ids).long().cuda()
            attention_mask = pb_utils.get_input_tensor_by_name(request, "attention_mask")
            attention_mask = attention_mask.as_numpy()
            attention_mask = torch.as_tensor(attention_mask).long().cuda()
            inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
            translation = self.model.generate(**inputs, num_beams=1)
            # Convert to numpy array on cpu:
            np_translation =  translation.cpu().int().detach().numpy()
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor(
                        "output",
                        np_translation.astype(self.output_dtype)
                    )
                ]
            )
            responses.append(inference_response)
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')
