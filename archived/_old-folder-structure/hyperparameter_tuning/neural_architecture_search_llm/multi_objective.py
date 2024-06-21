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


def get_pareto_optimal(costs: np.ndarray):
    """
    Find the pareto-optimal points
    :param costs: (n_points, m_cost_values) array
    :return: (n_points, 1) indicator if point is on pareto front or not.
    """
    assert type(costs) == np.ndarray
    assert costs.ndim == 2

    # first assume all points are pareto optimal
    is_pareto = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_pareto[i]:
            # determine all points that have a smaller cost
            all_with_lower_costs = np.any(costs < c, axis=1)
            keep_on_front = np.logical_and(all_with_lower_costs, is_pareto)
            is_pareto = keep_on_front
            is_pareto[i] = True  # keep self
    return is_pareto
