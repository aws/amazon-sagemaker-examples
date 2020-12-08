Get started with SageMaker Debugger
===================================

.. toctree::
   :maxdepth: 1

   mnist_tensor_analysis/mnist_tensor_analysis
   mnist_tensor_plot/mnist-tensor-plot


Real-time analysis
==================

.. toctree::
   :maxdepth: 1

   model_specific_realtime_analysis/autoencoder_mnist/autoencoder_mnist
   model_specific_realtime_analysis/bert_attention_head_view/bert_attention_head_view
   model_specific_realtime_analysis/cnn_class_activation_maps/cnn_class_activation_maps


MXNet
=================

.. toctree::
   :maxdepth: 1

   mxnet_realtime_analysis/mxnet-realtime-analysis
   mxnet_spot_training/mxnet-spot-training-with-sagemakerdebugger


PyTorch
=================

.. toctree::
   :maxdepth: 1

   pytorch_iterative_model_pruning/iterative_model_pruning_resnet
   pytorch_iterative_model_pruning/iterative_model_pruning_alexnet
   pytorch_custom_container/pytorch_byoc_smdebug


Profiling
^^^^^^^^^

.. toctree::
   :maxdepth: 1

   /sagemaker-debugger/pytorch_profiling/pt-resnet-profiling-single-gpu-single-node.ipynb
   /sagemaker-debugger/pytorch_profiling/pt-resnet-profiling-multi-gpu-single-node.ipynb
   /sagemaker-debugger/pytorch_profiling/pt-resnet-profiling-multi-gpu-multi-node


TensorFlow
====================

Tensorflow 2.x
--------------

.. toctree::
   :maxdepth: 1

   tensorflow2/tensorflow2_zero_code_change/tf2-keras-default-container
   .. tensorflow2/tensorflow2_keras_custom_container/tf2-keras-custom-container


Profiling
^^^^^^^^^

.. toctree::
   :maxdepth: 1

   /sagemaker-debugger/tensorflow_profiling/tf-resnet-profiling-single-gpu-single-node
   /sagemaker-debugger/tensorflow_profiling/tf-resnet-profiling-multi-gpu-multi-node
   .. /sagemaker-debugger/tensorflow_profiling/low_batch_size
   /sagemaker-debugger/tensorflow_profiling/dataset_bottleneck
   .. /sagemaker-debugger/tensorflow_profiling/callback_bottleneck


TensorFlow 1.x
--------------

.. toctree::
   :maxdepth: 1

   tensorflow_builtin_rule/tf-mnist-builtin-rule
   tensorflow_action_on_rule/detect_stalled_training_job_and_stop
   tensorflow_action_on_rule/tf-mnist-stop-training-job
   tensorflow_keras_custom_rule/tf-keras-custom-rule


XGBoost
=================

.. toctree::
   :maxdepth: 1

   xgboost_builtin_rules/xgboost-regression-debugger-rules
   xgboost_realtime_analysis/xgboost-realtime-analysis
   xgboost_census_explanations/xgboost-census-debugger-rules


Bring your own container
========================

.. toctree::
   :maxdepth: 1

   build_your_own_container_with_debugger/debugger_byoc
