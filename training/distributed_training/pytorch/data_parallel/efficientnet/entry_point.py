# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file
# except in compliance with the License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "LICENSE.txt" file accompanying this file. This file is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied. See the License for
# the specific language governing permissions and limitations under the License.

import subprocess
import sys
import os

exe = "python"

launcher = "/workspace/efficientnet/launch.py"

# Add the hyperparameters
cmd_list = [exe] + [launcher] + sys.argv[1:]
cmd = " ".join(cmd_list)

# Add the training data path. This is a data channel assigned to SM as the inputs parameter when calling estimator.fit()
cmd += " "
cmd += "--data " + os.environ["SM_CHANNEL_TRAIN"]

print("Final command is: ", cmd)

subprocess.run(cmd, shell=True)
