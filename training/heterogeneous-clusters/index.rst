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

 
You'll find TensorFlow (tf.data.service) and PyTorch (a gRPC based distributed data loading) examples on how to utilize Heterogeneous clusters in your training jobs. You can reuse these examples when enabling your own training workload to use heterogeneous clusters.
Try one of the notebooks:

.. admonition:: Frameworks

   - :ref:`hello-world-heterogeneous`
   - :ref:`tensorflow-heterogeneous`
   - :ref:`pytorch-heterogeneous`
   


.. _hello-world-heterogeneous:

Hello world
====================================

.. toctree::
   :maxdepth: 1

   hello.world.sagemaker/helloworld-example

.. _tensorflow-heterogeneous:

TensorFlow
====================================

.. toctree::
   :maxdepth: 1

   tf.data.service.sagemaker/hetero-tensorflow-restnet50
   

PyTorch
====================================

.. toctree::
   :maxdepth: 1

   pt.grpc.sagemaker/hetero-pytorch-mnist

