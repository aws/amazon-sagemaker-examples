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
from .utils import (
    register_mask_ffn,
    get_backbone,
    register_drop_layer,
    register_drop_attention_layer,
)


def get_ffn2(model, index):
    layer = get_layers(model)[index]
    ffn2 = layer.output
    return ffn2


def get_attention_output(model, index):
    layer = get_layers(model)[index]
    output = layer.attention
    return output


def get_layers(model):
    encoder = get_backbone(model).encoder
    layers = encoder.layer
    return layers


def mask_bert(model, neuron_mask, head_mask):
    num_hidden_layers = neuron_mask.shape[0]

    assert head_mask.shape[0] == num_hidden_layers

    handles = []
    for layer_idx in range(num_hidden_layers):
        ffn2 = get_ffn2(model, layer_idx)
        handle = register_mask_ffn(ffn2, neuron_mask[layer_idx])
        handles.append(handle)

        if neuron_mask[layer_idx].sum() == 0:
            handle = register_drop_layer(ffn2)
            handles.append(handle)

        if head_mask[layer_idx].sum() == 0:
            attention = get_attention_output(model, layer_idx)
            handle = register_drop_attention_layer(attention)

            handles.append(handle)

    return handles
