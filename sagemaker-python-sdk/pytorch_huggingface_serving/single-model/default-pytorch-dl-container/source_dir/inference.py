# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import logging
import json
import torch 

from transformers import BartTokenizer, BartForConditionalGeneration

logger = logging.getLogger(__name__)

TOKENIZER_PATH = '/opt/ml/model/bart_tokenizer/'
MODEL_PATH = '/opt/ml/model/bart_model/'

tokenizer = BartTokenizer.from_pretrained(TOKENIZER_PATH)

def model_fn(model_dir):
    device = get_device()
    model = BartForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)
    return model

def input_fn(json_request_data, content_type='application/json'):  
    input_data = json.loads(json_request_data)
    text_to_summarize = input_data['text']
    return text_to_summarize

def predict_fn(text_to_summarize, model):
    device = get_device()
    
    text_input_ids = tokenizer.batch_encode_plus([text_to_summarize], 
                                             return_tensors='pt',
                                             max_length=1024,
                                             return_token_type_ids=False, 
                                             return_attention_mask=False).to(device)

    summary_ids = model.generate(text_input_ids['input_ids'],
                                 num_beams=4,
                                 length_penalty=2.0,
                                 max_length=256,
                                 min_length=56,
                                 no_repeat_ngram_size=3)

    summary_txt = tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)

    return summary_txt
    
def output_fn(summary_txt, accept='application/json'):
    return json.dumps(summary_txt), accept

def get_device():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return device