import json
import numpy as np
import os
import sys
import triton_python_backend_utils as pb_utils

import torch
import torch_neuronx

from transformers_neuronx.opt.model import OPTForSampling

class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """
    
    def _validate_and_get_index(self, name):
        parts = name.split('__')
        if len(parts) != 2:
            raise pb_utils.TritonModelException(
                "tensor names are expected to be in format <name>__<index>, got {}"
                .format(name))

        if not parts[1].isnumeric():
            raise pb_utils.TritonModelException(
                "tensor names are expected to be in format <name>__<index> where <index> should be numeric, got {}"
                .format(name))

        return int(parts[1])

    def _validate_input_dict(self, expected_count):
        for i in range(expected_count):
            if i not in self.input_dict:
                raise pb_utils.TritonModelException(
                    "input corresponding to index {} not found".format(i))

    def _validate_output_dict(self, expected_count):
        for i in range(expected_count):
            if i not in self.output_dict:
                raise pb_utils.TritonModelException(
                    "output corresponding to index {} not found".format(i))

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.

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
        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args['model_config'])

        if (len(model_config['instance_group']) != 1):
            raise pb_utils.TritonModelException(
                "this model supports only a single instance group, got {}".
                format(len(model_config['instance_group'])))

        instance_group_config = model_config['instance_group'][0]
        instance_count = instance_group_config['count']

        instance_idx = 0
        if instance_count > 1:
            instance_name_parts = args['model_instance_name'].split("_")
            if not instance_name_parts[-1].isnumeric():
                raise pb_utils.TritonModelException(
                    "internal error: the model instance name should end with '_<instance_idx>', got {}"
                    .format(args['model_instance_name']))
            instance_idx = int(instance_name_parts[-1])

        params = model_config['parameters']
        compiled_model = params['COMPILED_MODEL']['string_value']
        compiled_model = os.path.join(args['model_repository'], compiled_model)

        nc_start_idx = int(params['NEURON_CORE_START_INDEX']['string_value'])
        nc_end_idx = int(params['NEURON_CORE_END_INDEX']['string_value'])
        if nc_end_idx < nc_start_idx:
            raise pb_utils.TritonModelException(
                "the neuron core end index should be greater than or equal to the start index"
            )

        threads_per_core = int(params['NUM_THREADS_PER_CORE']['string_value'])
        if threads_per_core < 1:
            raise pb_utils.TritonModelException(
                "the number of threads per core should be greater than or equal to 1"
            )
        num_threads = (nc_end_idx - nc_start_idx + 1) * threads_per_core

        total_core_count = nc_end_idx - nc_start_idx + 1
        if (instance_count > total_core_count):
            raise pb_utils.TritonModelException(
                "can not distribute {} triton model instances to {} neuron cores"
                .format(instance_count, total_core_count))
        cores_per_instance = total_core_count // instance_count

        self.input_dict = {}
        expected_input_count = 0
        for config_input in model_config['input']:
            index = self._validate_and_get_index(config_input['name'])
            self.input_dict[index] = [
                config_input['name'], config_input['data_type'],
                config_input['dims']
            ]
            expected_input_count += 1
        self._validate_input_dict(expected_input_count)

        self.output_dict = {}
        for config_output in model_config['output']:
            index = self._validate_and_get_index(config_output['name'])
            self.output_dict[index] = [
                config_output['name'], config_output['data_type'],
                config_output['dims']
            ]

        adjusted_nc_start_idx = (instance_idx *
                                 cores_per_instance) + nc_start_idx
        cores_range = '{}-{}'.format(
            adjusted_nc_start_idx,
            (adjusted_nc_start_idx + cores_per_instance - 1))
        os.environ["NEURON_RT_VISIBLE_CORES"] = cores_range

        consumed_cores_list = [i for i in range(cores_per_instance)]

        #self.model_neuron = torch.jit.load(compiled_model)
        batch_size = 1
        tp_degree = 12
        n_positions = 2048
        amp = 'bf16'
        unroll = None
        self.model_neuron = OPTForSampling.from_pretrained(compiled_model, batch_size=batch_size, amp=amp, tp_degree=tp_degree, n_positions=n_positions, unroll=unroll)
        self.model_neuron.to_neuron()

        self.model_neuron.num_workers = num_threads

    def execute(self, requests):
        """`execute` MUST be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse

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
        inputs = []
        num_requests = len(requests)
        request_batch_sizes = []
        for i in self.input_dict.keys():
            name, dt, shape = self.input_dict[i]
            first_tensor = torch.as_tensor(pb_utils.get_input_tensor_by_name(requests[0],
                                                            name).as_numpy())
            request_batch_sizes.append(first_tensor.size(dim=0))
            batched_tensor = first_tensor
            for j in range(1, num_requests):
                tensor = torch.as_tensor(pb_utils.get_input_tensor_by_name(requests[j],
                                                            name).as_numpy())
                request_batch_sizes.append(request_batch_sizes[-1] + tensor.size(dim=0))
                batched_tensor = torch.cat((batched_tensor, tensor), dim=0)
            inputs.append(batched_tensor)
        
        batched_results = self.model_neuron.sample(batched_tensor, 512)
        
        chunky_batched_results = []
        for i in self.output_dict.keys():
            batch = batched_results[i] if isinstance(batched_results, tuple) else batched_results
            chunky_batched_results.append(torch.tensor_split(batch, request_batch_sizes, dim=0))
        for i in range(num_requests):
            output_tensors = []
            for j in self.output_dict.keys():
                name, dt, shape = self.output_dict[j]
                result = chunky_batched_results[j][i]
                output_tensor = pb_utils.Tensor(
                    name, result.numpy().astype(
                        pb_utils.triton_string_to_numpy(dt)))
                output_tensors.append(output_tensor)
            inference_response = pb_utils.InferenceResponse(
                output_tensors=output_tensors)
            responses.append(inference_response)
    
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')

