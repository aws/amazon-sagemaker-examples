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
import numpy as np
import torch


class SearchSpace(object):
    """
    Setting the mask to 1 means we keep the corresponding head / unit
    """

    def __init__(self, config, rng=None):
        self.config = config

        if config.model_type == "gpt2":
            self.num_heads = config.n_head
            self.num_layers = config.n_layer
            self.intermediate_size = (
                config.n_inner if config.n_inner is not None else 4 * config.hidden_size
            )

        else:
            self.num_heads = config.num_attention_heads
            self.num_layers = config.num_hidden_layers
            self.intermediate_size = config.intermediate_size

        if rng is None:
            self.rng = np.random.RandomState(np.random.randint(2**32 - 1))
        else:
            self.rng = rng

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def get_smallest_sub_network(self):
        raise NotImplementedError


class SmallSearchSpace(SearchSpace):
    def __call__(self, *args, **kwargs):
        num_layers = self.rng.randint(self.num_layers)
        num_heads = self.rng.choice([int(self.num_heads / 2 ** i) for i in range(int(np.log2(self.num_heads)) + 1)])
        num_units = self.rng.randint(1, self.intermediate_size)

        return self._create_mask(num_layers, num_heads, num_units)

    def _create_mask(self, num_layers, num_heads, num_units):
        head_mask = torch.ones((self.num_layers, self.num_heads))
        ffn_mask = torch.ones((self.num_layers, self.intermediate_size))
        head_mask[num_layers:] = 0
        head_mask[:num_layers, num_heads:] = 0
        ffn_mask[num_layers:] = 0
        ffn_mask[:num_layers, num_units:] = 0
        return head_mask, ffn_mask

    def get_smallest_sub_network(self):
        num_layers = 1
        num_heads = 1
        num_units = 1

        return self._create_mask(num_layers, num_heads, num_units)
