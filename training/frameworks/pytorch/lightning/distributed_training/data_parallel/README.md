# Distributed data parallel training with PyTorch Lightning

Note: Refer to the following notebooks for setting up the Notebooks for launching distributed training.
* https://github.com/aws/amazon-sagemaker-examples/blob/main/training/distributed_training/pytorch/data_parallel/mnist/pytorch_smdataparallel_mnist_demo.ipynb
* https://github.com/aws/amazon-sagemaker-examples/blob/main/training/distributed_training/pytorch/data_parallel/bert/pytorch_smdataparallel_bert_demo.ipynb

Lightning uses the the concept of strategy, which is passed to the trainer object to configure the distributed training setup. 
https://pytorch-lightning.readthedocs.io/en/stable/extensions/strategy.html

The legacy way of configuring distributed training is to use plugins. 

This directory contains examples of both using strategy and plugin to configure distributed training.

Note that distributed trainings cannot be executed in an interactive environment such a Jupyter Notebook. Instead, the user must use the Sagemaker Training Toolkit which provides the abstraction to invoke distributed trainings on SageMaker. The invocation itself can be done through a Jupyter Notebook.

The directory is organized as follows

* `distributed_training/data_parallel` directory contains `ddp` and `smddp`.
* `ddp` uses pytorch distributed data parallel c10d implementation to execute distributed training. Ref: https://pytorch.org/docs/master/notes/ddp.html
* `smddp` uses the pytorch ddp implementation backed by SageMaker Distributed Data Parallel library, which provides accelerated implementation of the collectives used by ddp.
* Both `ddp` and `smddp` contain examples of multi node, multi device trainings using lightning plugin && strategy.