####################
Distributed Training
####################

The SageMaker built-in libraries of algorithms consists of 18 popular machine
learning algorithms.
Many of them were rewritten from scratch to be scalable and distributed out of
the box.
If you want to use distributed deep learning training code,
we recommend Amazon SageMaker’s distributed training libraries.
SageMaker’s distributed training libraries make it easier for you to write
highly scalable and cost-effective custom data parallel and model parallel deep
learning training jobs.

SageMaker distributed training libraries offer both data-parallel and
model-parallel training strategies. It combines software and hardware
technologies to improve inter-GPU and inter-node communications.
It extends SageMaker’s training capabilities with built-in options that require
only small code changes to your training scripts.

To learn how, try one of the notebooks in the following framework sections.

.. admonition:: Frameworks

   - :ref:`mxnet-distributed`
   - :ref:`pytorch-distributed`
   - :ref:`tensorflow-distributed`


SageMaker distributed data parallel
====================================

SageMaker distributed data parallel (SDP) extends SageMaker’s training
capabilities on deep learning models with near-linear scaling efficiency,
achieving fast time-to-train with minimal code changes.

SDP optimizes your training job for AWS network infrastructure and EC2 instance
topology.

SDP takes advantage of gradient update to communicate between nodes with a
custom AllReduce algorithm.

When training a model on a large amount of data, machine learning practitioners
will often turn to distributed training to reduce the time to train. In some
cases, where time is of the essence, the business requirement is to finish
training as quickly as possible or at least within a constrained time period.
Then, distributed training is scaled to use a cluster of multiple nodes, meaning
not just multiple GPUs in a computing instance, but multiple instances with
multiple GPUs. As the cluster size increases, so does the significant drop in
performance. This drop in performance is primarily caused the communications
overhead between nodes in a cluster.

SageMaker distributed (SMD) offers two options for distributed training:
SageMaker model parallel (SMP) and SageMaker data parallel (SDP). This guide
focuses on how to train models using a data parallel strategy.
For more information on training with a model parallel strategy,
refer to SageMaker distributed model parallel.

.. admonition:: More resources:

      - `SageMaker data parallel developer guide <https://docs.aws.amazon.com/sagemaker/latest/dg/data-parallel.html>`_
      - `SageMaker Python SDK - data parallel APIs <https://sagemaker.readthedocs.io/en/stable/api/training/smd_data_parallel.html>`_


SageMaker distributed model parallel
====================================

Amazon SageMaker distributed model parallel (SMP) is a model parallelism library
for training large deep learning models that were previously difficult to train
due to GPU memory limitations. SMP automatically and efficiently splits a model
across multiple GPUs and instances and coordinates model training, allowing you
to increase prediction accuracy by creating larger models with more parameters.

You can use SMP to automatically partition your existing TensorFlow and PyTorch
workloads across multiple GPUs with minimal code changes. The SMP API can be
accessed through the Amazon SageMaker SDK.

Use the following sections to learn more about the model parallelism and the SMP
library.

.. admonition:: More resources:

      - `SageMaker model parallel developer guide <https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel.html>`_
      - `SageMaker Python SDK - model parallel APIs <https://sagemaker.readthedocs.io/en/stable/api/training/smd_model_parallel.html>`_


.. _pytorch-distributed:

PyTorch
====================================

SageMaker distributed data parallel (SDP)
-----------------------------------------

.. toctree::
   :maxdepth: 1

   pytorch/data_parallel/bert/pytorch_smdataparallel_bert_demo
   pytorch/data_parallel/maskrcnn/pytorch_smdataparallel_maskrcnn_demo
   pytorch/data_parallel/mnist/pytorch_smdataparallel_mnist_demo


SageMaker distributed model parallel (SMP)
-----------------------------------------

.. toctree::
   :maxdepth: 1

   pytorch/model_parallel/bert/smp_bert_tutorial
   pytorch/model_parallel/mnist/pytorch_smmodelparallel_mnist


Horovod
-------

.. toctree::
   :maxdepth: 1

   /sagemaker-python-sdk/pytorch_horovod_mnist/pytorch_mnist_horovod


.. _tensorflow-distributed:

TensorFlow2
====================================

SageMaker distributed data parallel (SDP)
-----------------------------------------

.. toctree::
   :maxdepth: 1

   tensorflow/data_parallel/bert/tensorflow2_smdataparallel_bert_demo
   tensorflow/data_parallel/maskrcnn/tensorflow2_smdataparallel_maskrcnn_demo
   tensorflow/data_parallel/mnist/tensorflow2_smdataparallel_mnist_demo


SageMaker distributed model parallel (SMP)
-----------------------------------------

.. toctree::
   :maxdepth: 1

   tensorflow/model_parallel/mnist/tensorflow_smmodelparallel_mnist


Horovod
-------

.. toctree::
   :maxdepth: 1

   /sagemaker-python-sdk/keras_script_mode_pipe_mode_horovod/tensorflow_keras_CIFAR10
   /advanced_functionality/distributed_tensorflow_mask_rcnn/mask-rcnn-s3


.. _mxnet-distributed:

Apache MXNet
====================================

Horovod
-------

.. toctree::
   :maxdepth: 1

   /sagemaker-python-sdk/mxnet_mnist/mxnet_mnist
   /sagemaker-python-sdk/mxnet_horovod_fasterrcnn/horovod_deployment_notebook
   /sagemaker-python-sdk/mxnet_horovod_maskrcnn/horovod_deployment_notebook
   /sagemaker-python-sdk/mxnet_horovod_mnist/mxnet_mnist_horovod


In addition to the notebook, this topic is covered in this workshop topic:
`Parallelized data distribution (sharding) <https://sagemaker-workshop.com/builtin/parallelized.html>`_
