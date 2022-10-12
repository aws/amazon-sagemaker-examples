import json
import logging
import numpy as np
import subprocess
import sys
import os

import triton_python_backend_utils as pb_utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TritonPythonModel:
    """This model loops through different dtypes to make sure that
    serialize_byte_tensor works correctly in the Python backend.
    """
    
    def __mean_pooling(self, token_embeddings, attention_mask):
        logger.info("token_embeddings: {}".format(token_embeddings))
        logger.info("attention_mask: {}".format(attention_mask))
        
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def initialize(self, args):
        self.model_dir = args['model_repository']
        subprocess.check_call([sys.executable, "-m", "pip", "install", '-r', f'{self.model_dir}/requirements.txt'])
        global torch
        import torch 
        
        self.device_id = args['model_instance_device_id']
        self.model_config = model_config = json.loads(args['model_config'])
        self.device = torch.device(f'cuda:{self.device_id}') if torch.cuda.is_available() else torch.device('cpu')

        output0_config = pb_utils.get_output_config_by_name(
            model_config, "SENT_EMBED")
        
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config["data_type"])
        
    def execute(self, requests):

        responses = []
        
        for request in requests:
            tok_embeds = pb_utils.get_input_tensor_by_name(request, "TOKEN_EMBEDS_POST")
            attn_mask = pb_utils.get_input_tensor_by_name(request, "ATTENTION_POST")
            
            tok_embeds = tok_embeds.as_numpy()
            
            logger.info("tok_embeds: {}".format(tok_embeds))
            logger.info("tok_embeds shape: {}".format(tok_embeds.shape))
            
            tok_embeds = torch.tensor(tok_embeds,device=self.device)
            
            logger.info("tok_embeds_tensor: {}".format(tok_embeds))
            
            attn_mask = attn_mask.as_numpy()
            
            logger.info("attn_mask: {}".format(attn_mask))
            logger.info("attn_mask shape: {}".format(attn_mask.shape))
            
            attn_mask = torch.tensor(attn_mask,device=self.device)
            
            logger.info("attn_mask_tensor: {}".format(attn_mask))
            
            sentence_embeddings = self.__mean_pooling(tok_embeds, attn_mask)
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

            out_0 = np.array(sentence_embeddings.cpu(),dtype=self.output0_dtype)
            logger.info("out_0: {}".format(out_0))
            
            out_tensor_0 = pb_utils.Tensor("SENT_EMBED", out_0)
            logger.info("out_tensor_0: {}".format(out_tensor_0))
            
            responses.append(pb_utils.InferenceResponse([out_tensor_0]))
        
        return responses