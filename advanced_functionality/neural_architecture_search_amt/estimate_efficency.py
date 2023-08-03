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
import torch
import numpy as np


def mac_per_head(
    seq_len,
    hidden_size,
    attention_head_size,
):
    per_head_qkv = lambda seq_len: 3 * seq_len * hidden_size * attention_head_size
    per_head_attn = lambda seq_len: 2 * seq_len * seq_len * attention_head_size
    per_head_output = lambda seq_len: seq_len * attention_head_size * hidden_size
    mac = per_head_qkv(seq_len) + per_head_attn(seq_len) + per_head_output(seq_len)
    return mac


def mac_per_neuron(seq_len, hidden_size):
    return 2 * seq_len * hidden_size


def compute_mac(
    num_heads_per_layer,
    num_neurons_per_layer,
    seq_len,
    hidden_size,
    attention_head_size,
):
    mac = 0.0
    for num_heads, num_neurons in zip(num_heads_per_layer, num_neurons_per_layer):
        attention_mac = num_heads * mac_per_head(
            seq_len, hidden_size, attention_head_size
        )
        ffn_mac = num_neurons * mac_per_neuron(seq_len, hidden_size)
        mac += attention_mac + ffn_mac
    return mac


def compute_parameters(dmodel, dhead, num_heads_per_layer, num_neurons_per_layer):

    num_layers = num_heads_per_layer.shape[0]
    assert num_layers == num_neurons_per_layer.shape[0]

    num_parameters = 0
    for layer in range(num_layers):
        n_layer_norm = 2 * dmodel
        if num_heads_per_layer[layer] > 0:
            n_attention = (
                (dmodel * dhead + dhead) * num_heads_per_layer[layer] * 3
            )  # attention
            n_attention += dmodel * dmodel + dmodel  # output
            n_attention += n_layer_norm
        else:
            n_attention = 0
        if num_neurons_per_layer[layer] > 0:
            n_ffn = (
                2 * dmodel * num_neurons_per_layer[layer]
                + dmodel
                + num_neurons_per_layer[layer]
            )
            n_ffn += n_layer_norm
        else:
            n_ffn = 0

        num_parameters += n_attention + n_ffn
    return int(num_parameters)


def compute_latency(model, tokenizer, batch, device):
    # train_dataset[0][sentence1_key],
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    repetitions = 300
    timings = np.zeros((repetitions, 1))
    # warm-up GPU
    for _ in range(10):
        _ = model(**tokenizer(batch, return_tensors="pt").to(device))
    # measure latency
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(**tokenizer(batch, return_tensors="pt").to(device))
            ender.record()
            # synchronize GPU
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    mean_syn = np.sum(timings) / repetitions

    return mean_syn
