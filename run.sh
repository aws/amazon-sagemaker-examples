#!/usr/bin/env bash

# Check to see if input has been provided:
if [ -z "$1" ]; then
    echo "Please provide the training dataset config file"
    echo "For example: training-data-configuration.json"
    exit 1
fi

cd /opt/ml/code/

config=$1

if [ -z "$2" ]; then
  echo "python3 load.py --sagemaker True --dataset_spec $config"
  python3 load.py --sagemaker True --dataset_spec $config
else
  if [ $2 == "--incremental" ]; then
    echo "python3 load.py --sagemaker True --dataset_spec $config --incremental whole"
    python3 load.py --sagemaker True --dataset_spec $config --incremental whole
  else
    model=$2
    if [ -z "$3" ]; then
      echo "python3 load.py --sagemaker True --dataset_spec $config --model_type $model"
      python3 load.py --sagemaker True --dataset_spec $config --model_type $model
    else
      echo "python3 load.py --sagemaker True --dataset_spec $config --model_type $model --incremental whole"
      python3 load.py --sagemaker True --dataset_spec $config --model_type $model --incremental whole
    fi
  fi
fi