import json
import logging
import numpy as np 
import subprocess
import sys

import triton_python_backend_utils as pb_utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TritonPythonModel:
    """This model loops through different dtypes to make sure that
    serialize_byte_tensor works correctly in the Python backend.
    """

    def initialize(self, args):
        self.model_dir = args['model_repository']
        subprocess.check_call([sys.executable, "-m", "pip", "install", '-r', f'{self.model_dir}/requirements.txt'])
        global transformers
        import transformers

        self.tokenizer = transformers.AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.model_config = model_config = json.loads(args['model_config'])

        output0_config = pb_utils.get_output_config_by_name(
            model_config, "OUTPUT0")
        output1_config = pb_utils.get_output_config_by_name(
            model_config, "OUTPUT1")
        
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])
        self.output1_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])
        
    def execute(self, requests):
        
        file = open("logs.txt", "w")
        
        responses = []
        for request in requests:
            logger.info("Request: {}".format(request))
            
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            in_0 = in_0.as_numpy()
            
            logger.info("in_0: {}".format(in_0))
                        
            tok_batch = []
            
            for i in range(in_0.shape[0]):                
                decoded_object = in_0[i,0].decode()
                
                logger.info("decoded_object: {}".format(decoded_object))
                                                
                tok_batch.append(decoded_object)
                
            logger.info("tok_batch: {}".format(tok_batch))
            
            tok_sent = self.tokenizer(tok_batch,
                                      padding='max_length',
                                      max_length=128,
                                     )

            
            logger.info("Tokens: {}".format(tok_sent))
            
            out_0 = np.array(tok_sent['input_ids'],dtype=self.output0_dtype)
            out_1 = np.array(tok_sent['attention_mask'],dtype=self.output1_dtype)
            out_tensor_0 = pb_utils.Tensor("OUTPUT0", out_0)
            out_tensor_1 = pb_utils.Tensor("OUTPUT1", out_1)

            responses.append(pb_utils.InferenceResponse([out_tensor_0,out_tensor_1]))
        return responses
