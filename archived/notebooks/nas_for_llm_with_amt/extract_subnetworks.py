# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
from collections import OrderedDict

import torch.nn as nn

from transformers import AutoModelForSequenceClassification
from transformers.models.bert.modeling_bert import BertForSequenceClassification, BertForMultipleChoice, BertConfig


def copy_linear_layer(new_layer, old_layer, weight_shape, bias_shape):
    old_state = old_layer.state_dict()
    new_state_dict = OrderedDict()
    new_state_dict["weight"] = old_state["weight"][: weight_shape[0], : weight_shape[1]]
    new_state_dict["bias"] = old_state["bias"][:bias_shape]
    new_layer.load_state_dict(new_state_dict)


def copy_layer_norm(new_layer, old_layer):
    old_state = old_layer.state_dict()
    new_state_dict = OrderedDict()
    new_state_dict["weight"] = old_state["weight"]
    new_state_dict["bias"] = old_state["bias"]
    new_layer.load_state_dict(new_state_dict)


def get_final_bert_model(original_model, new_model_config):
    assert isinstance(original_model, (BertForSequenceClassification,
                                       AutoModelForSequenceClassification,
                                       BertForMultipleChoice)), f"Make sure to pass a valid BERT model for" \
                                                                f" sequence classification or multiple choice Q/A"

    assert isinstance(new_model_config, BertConfig), f"Make sure to pass a valid BERT model for" \
                                                                f" sequence classification or multiple choice Q/A"

    original_model.eval()
    new_model = AutoModelForSequenceClassification.from_config(new_model_config)
    new_model.eval()

    new_model.bert.embeddings.load_state_dict(
        original_model.bert.embeddings.state_dict()
    )
    new_model.bert.pooler.load_state_dict(original_model.bert.pooler.state_dict())
    new_model.classifier.load_state_dict(original_model.classifier.state_dict())

    num_attention_heads = new_model_config.num_attention_heads
    attention_head_size = new_model_config.attention_head_size
    all_head_size = num_attention_heads * attention_head_size
    for li, layer in enumerate(new_model.bert.encoder.layer):
        attention = layer.attention
        attention.self.query = nn.Linear(new_model_config.hidden_size, all_head_size)
        attention.self.key = nn.Linear(new_model_config.hidden_size, all_head_size)
        attention.self.value = nn.Linear(new_model_config.hidden_size, all_head_size)
        attention.output.dense = nn.Linear(
            all_head_size,
            new_model_config.hidden_size,
        )

        attention.self.all_head_size = all_head_size
        attention.self.attention_head_size = attention_head_size

        mha_original_model = original_model.bert.encoder.layer[li].attention
        copy_linear_layer(
            attention.self.query,
            mha_original_model.self.query,
            (all_head_size, new_model_config.hidden_size),
            (all_head_size),
        )

        copy_linear_layer(
            attention.self.key,
            mha_original_model.self.key,
            (all_head_size, new_model_config.hidden_size),
            (all_head_size),
        )

        copy_linear_layer(
            attention.self.value,
            mha_original_model.self.value,
            (all_head_size, new_model_config.hidden_size),
            (all_head_size),
        )

        copy_linear_layer(
            attention.output.dense,
            mha_original_model.output.dense,
            (new_model_config.hidden_size, all_head_size),
            (new_model_config.hidden_size),
        )

        copy_layer_norm(attention.output.LayerNorm, mha_original_model.output.LayerNorm)

        ffn_layer = layer.intermediate.dense
        ffn_original_model = original_model.bert.encoder.layer[li].intermediate.dense
        copy_linear_layer(
            ffn_layer,
            ffn_original_model,
            (new_model_config.intermediate_size, new_model_config.hidden_size),
            (new_model_config.intermediate_size),
        )

        ffn_layer = layer.output.dense
        ffn_original_model = original_model.bert.encoder.layer[li].output.dense
        copy_linear_layer(
            ffn_layer,
            ffn_original_model,
            (new_model_config.hidden_size, new_model_config.intermediate_size),
            (new_model_config.hidden_size),
        )

        copy_layer_norm(
            layer.output.LayerNorm,
            original_model.bert.encoder.layer[li].output.LayerNorm,
        )

    return new_model
