####################
Heterogeneous Clusters
####################

SageMaker Training Heterogeneous Clusters allows you to run one training job 
that includes instances of different types. For example a GPU instance like 
ml.p4d.24xlarge and a CPU instance like c5.18xlarge. 

One primary use case is offloading CPU intensive tasks like image 
pre-processing (data augmentation) from the GPU instance to a dedicate 
CPU instance, so you can fully utilize the expensive GPUs, and arrive at 
an improved time and cost to train.

.. admonition:: More resources:

      - `SageMaker heterogeneous cluster developer guide <https://docs.aws.amazon.com/sagemaker/latest/dg/train-heterogeneous-cluster.html>`_

 
See the following example notebooks:

Hello World
====================================
This minimal example launches a Heterogeneous cluster training job, print environment information, and exit.

.. toctree::
   :maxdepth: 1

   hello.world.sagemaker/helloworld-example


TensorFlow
====================================
This example is a reusable implementation of Heterogeneous cluster with TensorFlow's tf.data.service

.. toctree::
   :maxdepth: 1

   tf.data.service.sagemaker/hetero-tensorflow-restnet50
   

PyTorch
====================================
This example is a reusable implementation of Heterogeneous cluster with gRPC based data loader

.. toctree::
   :maxdepth: 1

   pt.grpc.sagemaker/hetero-pytorch-mnist

