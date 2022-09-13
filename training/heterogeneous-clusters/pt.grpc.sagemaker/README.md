# SageMaker heterogeneous Training ("Hetero") - Pytorch example (MNIST Dataset)
This example demonstrates a more general way of offloading pre-processing to auxiliary devices using gRPC, the same protocol underlying the TensorFlow data service. We use here pytorch 1.11 framework. The job is submitted to SageMaker using Hetero feature that allows you to run one training job that includes instances of different types (for example a GPU instance like ml.g5.2xlarge and a CPU instance like c5n.9xlarge). The primary use case here is offloading CPU intensive tasks like image pre-processing (data augmentation) from the GPU instance to a dedicate CPU instance, so you can fully utilize the exensive GPU, and arrive at an improved time and cost to train.
 

## Instructions
Follow steps in [notebook](./hetero-pytorch-mnist.ipynb)