#################
Training Compiler
#################

Train deep learning (DL) models faster on scalable GPU instances managed by SageMaker.


Get started with SageMaker Training Compiler
============================================

PyTorch with Hugging Face Transformers
--------------------------------------

For single-node single-GPU training:

.. toctree::
   :maxdepth: 1

   huggingface/pytorch_single_gpu_single_node/albert-base-v2/albert-base-v2
   huggingface/pytorch_single_gpu_single_node/bert-base-cased/bert-base-cased-single-node-single-gpu
   huggingface/pytorch_single_gpu_single_node/roberta-base/roberta-base
   tensorflow/single_gpu_single_node/vision-transformer

For single-node multi-GPU training:

.. toctree::
   :maxdepth: 1

   huggingface/pytorch_multiple_gpu_single_node/language-modeling-multi-gpu-single-node
   tensorflow/multiple_gpu_single_node/vision-transformer

For multi-node multi-GPU training:

.. toctree::
   :maxdepth: 1

   huggingface/pytorch_multiple_gpu_multiple_node/language-modeling-multi-gpu-multi-node


For more information, see `Amazon SageMaker Training Compiler <https://docs.aws.amazon.com/sagemaker/latest/dg/training-compiler.html>`__
in the Amazon SageMaker Developer Guide.
