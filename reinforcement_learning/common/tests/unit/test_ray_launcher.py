# Copyright 2017-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import

import pytest
from mock import MagicMock, Mock, patch
from sagemaker_rl.ray_launcher import SageMakerRayLauncher


@patch("sagemaker_rl.ray_launcher.SageMakerRayLauncher.__init__", return_value=None)
@patch("sagemaker_rl.ray_launcher.change_permissions_recursive")
def test_pytorch_save_checkpoint_and_serving_model(change_permission, launcher_init):
    launcher = SageMakerRayLauncher()
    launcher.copy_checkpoints_to_model_output = Mock()
    launcher.create_tf_serving_model = Mock()
    launcher.save_experiment_config = Mock()

    launcher.save_checkpoint_and_serving_model(use_pytorch=True)
    launcher.create_tf_serving_model.assert_not_called()
    launcher.save_checkpoint_and_serving_model(use_pytorch=False)
    launcher.create_tf_serving_model.assert_called_once()
    assert 4 == change_permission.call_count
