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

# This is the fairseq training launcher
exe = 'fairseq-train'

# os.environ['SM_CHANNEL_TRAIN'] is the root directory of the fsx mount, we will attach
# the actual training dataset directory to it and pass to fairseq-train
data_dir = os.environ['SM_CHANNEL_TRAIN'] + '<dataset_dir>'

# Generate the full command
cmd_list = [exe] + [data_dir] + sys.argv[1:]
cmd = ' '.join(cmd_list)

# Attach one more parameter to the command. Here os.environ['SM_MODEL_DIR'] is the root
# of the output directory of SageMaker. We want fairseq to save model checkpoints to here
cmd += ' --save-dir ' + os.environ['SM_MODEL_DIR']

print('Final command is: ', cmd)

# Invoke the command
subprocess.run(cmd, shell=True)