#!/bin/bash

export SM_HOSTS='0'
export SM_CURRENT_HOST='0'
export SM_MODEL_DIR='/tmp/model'
export SM_CHANNEL_TRAINING='/tmp/data'
export SM_NUM_GPUS='0'

python train.py --epoch 0 --debug
