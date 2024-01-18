# Amazon SageMaker Examples

## Remote Function

This repository contains example notebooks and scripts that show how to [use remote function to run your Machine Learning code as training jobs in SageMaker](https://docs.aws.amazon.com/sagemaker/latest/dg/train-remote-decorator.html).

* [Quick Start notebook](quick_start/quick_start.ipynb) contains hands-on guide to run sample code as training jobs in SageMaker via remote function feature.
* [PyTorch MNSIT notebook](pytorch_mnist_sample_notebook/pytorch_mnist.ipynb) shows the adaption of an existing SageMaker example to use remote function feature.
* [PyTorch MNSIT script](pytorch_mnist_sample_script/) shows how to use remote function feature in Python scripts instead of notebooks.
* [HuggingFace notebook](huggingface_text_classification/huggingface.ipynb) shows how to run a text classification code using HuggingFace via remote function feature.
* [XGBoost notebook](xgboost_abalone/xgboost_abalone.ipynb) shows how to run a regression code using XGBoost via remote function feature.
