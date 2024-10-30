import json
import os

import torch
from djl_python import Input, Output, PairList
from djl_python.transformers_neuronx import TransformersNeuronXService
from djl_python.test_model import decode_encoded_output_binary, create_concurrent_batch_request
import numpy as np
import time
import asyncio, threading, copy

import triton_python_backend_utils as pb_utils

_MODEL_ARGS_FILENAME = "model.json"

class TritonPythonModel:

  def initialize(self, args):
    self.logger = pb_utils.Logger
    self.model_config = json.loads(args["model_config"])
    text_output_config = pb_utils.get_output_config_by_name(self.model_config, "text_output")
    self.text_output_dtype = pb_utils.triton_string_to_numpy(text_output_config["data_type"])
    self.__tasks = set()
    self._init_service()
    self.__tasks_inited = False
      

  @staticmethod
  def auto_complete_config(auto_complete_model_config):
      
    inputs = [
    {"name": "text_input", "data_type": "TYPE_STRING", "dims": [1]},
    {
        "name": "sampling_parameters",
        "data_type": "TYPE_STRING",
        "dims": [1],
        "optional": True,
    }
    ]
    outputs = [{"name": "text_output", "data_type": "TYPE_STRING", "dims": [-1]}]

    config = auto_complete_model_config.as_dict()
    input_names = []
    output_names = []
    for input in config['input']:
      input_names.append(input['name'])
    for output in config['output']:
      output_names.append(output['name'])

    for input in inputs:
      if input['name'] not in input_names:
          auto_complete_model_config.add_input(input)
    for output in outputs:
      if output['name'] not in output_names:
          auto_complete_model_config.add_output(output)

    auto_complete_model_config.set_model_transaction_policy(dict(decoupled=True))
    auto_complete_model_config.set_max_batch_size(0)

    return auto_complete_model_config

  def _init_service(self):
    self.logger.log_info("Enter: _init_service")

    max_batch_size = int(self.model_config.get('max_batch_size', 0))
    assert (max_batch_size == 0), "max_batch_size must be 0"
    self.using_decoupled = pb_utils.using_decoupled_model_transaction_policy(self.model_config) 
    assert (
        self.using_decoupled 
    ), "Python backend must be configured to use decoupled model transaction policy"

    model_args_filepath = os.path.join( 
        pb_utils.get_model_dir(), _MODEL_ARGS_FILENAME
    )
    assert os.path.isfile(
        model_args_filepath
    ), f"'{_MODEL_ARGS_FILENAME}' containing model args must be provided in '{pb_utils.get_model_dir()}'"
    with open(model_args_filepath) as file:
      self.model_args = json.load(file)

    self.batch_size = self.model_args.get("batch_size", None)
    if self.batch_size is None:
      self.batch_size = self.model_args.get("max_rolling_batch_size", 1)

    self.logger.log_info(f"initialize service: {self.model_args}")
    self.__service = TransformersNeuronXService()
    self.__service.initialize(self.model_args)
    self.logger.log_info("service initialized.")

    self.logger.log_info("Create request asyncio queue: maxsize {self.batch_size}")
    self.__request_queue = asyncio.Queue(maxsize=self.batch_size)

    self.logger.log_info("Create response asyncio queue: maxsize {self.batch_size}")
    self.__response_queue = asyncio.Queue(maxsize=self.batch_size)

    self.logger.log_info("Exit: _init_service")

  def get_sampling_params_dict(self, params_json):              
    params_dict = json.loads(params_json) if params_json else {}

    float_keys = [
        "temperature",
        "top_p"
    ]
    for k in float_keys:
        if k in params_dict:
            params_dict[k] = float(params_dict[k])
        
    int_keys = ["sequence_length", "top_k"]
    for k in int_keys:
        if k in params_dict:
            params_dict[k] = int(params_dict[k])

    if not params_dict:
        params_dict["max_new_tokens"] = 2048
        params_dict["top_k"] = 50
    else:
        if "max_new_tokens" not in params_dict:
          params_dict["max_new_tokens"] = 2048

    return params_dict

  async def __init_tasks(self):
    self.logger.log_info("Start respond loop")
    task = asyncio.create_task(self.__respond_loop())
    self.__tasks.add(task)
    task.add_done_callback(self.__tasks.discard)
    
    self.logger.log_info("Start generate loop")
    task = asyncio.create_task(self.__generate_loop())
    self.__tasks.add(task)
    task.add_done_callback(self.__tasks.discard)

    self.__tasks_inited = True

  async def execute(self, requests):
    if not self.__tasks_inited:
      try:
        await self.__init_tasks()
      except KeyError:
        print("Future not found or has already completed.")

    for request in requests:
      try:
        await self.__request_queue.put(request)
      except KeyError:
        print("Future not found or has already completed.")

  async def __check_new_requests(self):
    self.__new_requests = []
    if len(self.__requests) == 0:
      new_request = await self.__request_queue.get()
      self.__request_queue.task_done()
      self.__new_requests.append(new_request)
      self.__requests.append(new_request)

    while len(self.__requests) < self.batch_size:
      try:
        await asyncio.sleep(.001)
        new_request = self.__request_queue.get_nowait()
        self.__request_queue.task_done()
        self.__requests.append(new_request)
        self.__new_requests.append(new_request)
      except asyncio.QueueEmpty:
        break

  def __inference(self):

    for request in self.__new_requests:
      prompts = pb_utils.get_input_tensor_by_name(request, "text_input").as_numpy().flatten().tolist()
      inputs = [ p.decode("utf-8") if isinstance(p, bytes) else p for p in prompts]
      inputs = inputs[0] if len(inputs) == 1 else inputs

      parameters_input_tensor = pb_utils.get_input_tensor_by_name(request, "sampling_parameters")
      if parameters_input_tensor:
        parameters = parameters_input_tensor.as_numpy().flatten()
        parameters = parameters.tolist()[0] # assume uniform sampling parameters in batch
        parameters = parameters.decode('utf-8') if isinstance(parameters, bytes) else parameters
      else:
        parameters = request.parameters()
      params = self.get_sampling_params_dict(parameters)
      
      self.__input_list.append({"inputs": inputs,"parameters": params})

    n_new_requests = len(self.__new_requests)
    if n_new_requests > 0:
      properties = [{"eula": "true", "Content-type": "application/json"}]*len(self.__input_list)
      input = create_concurrent_batch_request(inputs=self.__input_list,
                                    properties=properties,
                                    serving_properties=copy.deepcopy(self.model_args))
      output = self.__service.inference(input)
      for i in range(output.content.size()):
        result = decode_encoded_output_binary(output.content.value_at(i))
        if i < len(self.__results):
          self.__results[i] = result
        else:
          self.__results.append(result)
    elif self.__service.rolling_batch: 
      self.__results = self.__service.rolling_batch.inference([])
    
  async def __generate_loop(self):
    
  
    while True:
      try:
        await self.__check_new_requests()

        unique_reqs = { f"{x}" for x in self.__requests }
        assert len(unique_reqs) == len(self.__requests), \
          f"requests are not unique, {self.__requests}"

        self.__inference()

        assert len(self.__requests) == len(self.__input_list), \
          f"requests: {len(self.__requests)} != input_list: {len(self.__input_list)}"

        assert len(self.__requests) == len(self.__results), \
          f"requests: {len(self.__requests)} != results: {len(self.__results)}"

        finished = []
        for i in range(len(self.__requests)):
          res = self.__results[i]
          if str(res['last']).lower() == 'true':
            req = self.__requests[i]
            input = self.__input_list[i]
            try:
              self.__response_queue.put_nowait((req, res))
            except asyncio.QueueFull:
              self.logger.log_info("response queue is full; await put")
              await self.__response_queue.put((req, res))
            finished.append((req, res, input))
          
        for item in finished:
          req, res, input = item
          self.__requests.remove(req)
          self.__results.remove(res)
          self.__input_list.remove(input)

        assert len(self.__requests) == len(self.__input_list), \
          f"requests: {len(self.__requests)} != input_list: {len(self.__input_list)}"

        assert len(self.__requests) == len(self.__results), \
          f"requests: {len(self.__requests)} != results: {len(self.__results)}"

      except Exception as e:
        self.logger.log_error(f"Unpexpected error: {e}. Inflight requests discarded. Reset engine.")
        self.reset()

  def reset(self):
    self.__requests = []
    self.__results = []
    self.__input_list = []

    if self.__service.rolling_batch:
      self.__service.rolling_batch.reset()

  async def __respond_loop(self):
    self.reset()

    while True:
      try:
        req, res = await self.__response_queue.get()
        self.__response_queue.task_done()
        
        t = threading.Thread(target=self.__send_response, 
          kwargs={"request": req, "response": res})
        t.start()
      except Exception as e:
        print(f"Error: respond loop exception {e}")

  def __send_response(self, request, response: dict):
    try:
      response_sender = request.get_response_sender()
      text_output = response['data']
    
      out_tensor = pb_utils.Tensor("text_output", np.array(text_output).astype(self.text_output_dtype))
      inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor])

      response_sender.send(inference_response, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)
    except Exception as e:
      print(f"send error: {request}", flush=True)
      print(e, flush=True)

  def finalize(self):
    self.logger.log_info("Cleaning up...")
    