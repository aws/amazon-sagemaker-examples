#!/bin/bash
source activate rapids
if [[ "$1" == "serve" ]]; then
    echo -e "@ entrypoint -> launching serving script \n"
    python serve.py
else
    echo -e "@ entrypoint -> launching training script \n"
    python train.py
fi