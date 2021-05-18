########
Debugger
########

Examples on how to use SageMaker Debugger.


Get started with SageMaker Debugger
===================================

Debugging
---------

.. toctree::
   :maxdepth: 1

   mxnet_realtime_analysis/mxnet-realtime-analysis
   xgboost_training_report/higgs_boson_detection

Profiling
---------

.. toctree::
   :maxdepth: 1

   debugger_interactive_analysis_profiling/interactive_analysis_profiling_data
   tensorflow_nlp_sentiment_analysis/sentiment-analysis-tf-distributed-training-bringyourownscript

----

Debugging Model Parameters
==========================

You can track and debug model parameters, such as weights, gradients, biases,
and scalar values of your training job.
Available deep learning frameworks are Apache MXNet, TensorFlow, PyTorch, and XGBoost.

Real-time analysis of deep learning models
------------------------------------------

.. toctree::
   :maxdepth: 1

   model_specific_realtime_analysis/autoencoder_mnist/autoencoder_mnist
   model_specific_realtime_analysis/bert_attention_head_view/bert_attention_head_view
   model_specific_realtime_analysis/cnn_class_activation_maps/cnn_class_activation_maps


Apache MXNet
------------

.. toctree::
   :maxdepth: 1

   mxnet_spot_training/mxnet-spot-training-with-sagemakerdebugger
   mnist_tensor_analysis/mnist_tensor_analysis
   mnist_tensor_plot/mnist-tensor-plot


TensorFlow 2.x
----------

.. toctree::
   :maxdepth: 1

   tensorflow2/tensorflow2_zero_code_change/tf2-keras-default-container
   .. tensorflow2/tensorflow2_keras_custom_container/tf2-keras-custom-container


TensorFlow 1.x
--------------

.. toctree::
  :maxdepth: 1

  tensorflow_builtin_rule/tf-mnist-builtin-rule
  tensorflow_action_on_rule/detect_stalled_training_job_and_stop
  tensorflow_action_on_rule/tf-mnist-stop-training-job
  tensorflow_keras_custom_rule/tf-keras-custom-rule


PyTorch
-------

.. toctree::
   :maxdepth: 1

   pytorch_iterative_model_pruning/iterative_model_pruning_resnet
   pytorch_iterative_model_pruning/iterative_model_pruning_alexnet
   pytorch_custom_container/pytorch_byoc_smdebug


XGBoost
-------

.. toctree::
    :maxdepth: 1

    xgboost_builtin_rules/xgboost-regression-debugger-rules
    xgboost_realtime_analysis/xgboost-realtime-analysis
    xgboost_census_explanations/xgboost-census-debugger-rules
    xgboost_training_report/higgs_boson_detection


Bring your own container
------------------------

.. toctree::
    :maxdepth: 1

    build_your_own_container_with_debugger/debugger_byoc

----

Profiling System Bottlenecks and Framework Operators
====================================================

Debugger provides the following profile features:

- **Monitoring system bottlenecks** – Monitor system resource utilization rate,
  such as CPU, GPU, memories, network, and data I/O metrics.
  This is a framework and model agnostic feature and available for
  any training jobs in SageMaker.
- **Profiling deep learning framework operations** – Profile deep learning operations
  of the TensorFlow and PyTorch frameworks, such as step durations, data loaders,
  forward and backward operations, Python profiling metrics, and framework-specific
  metrics.

Tensorflow
----------

.. toctree::
   :maxdepth: 1

   tensorflow_profiling/tf-resnet-profiling-single-gpu-single-node
   tensorflow_profiling/tf-resnet-profiling-multi-gpu-multi-node
   tensorflow_profiling/tf-resnet-profiling-multi-gpu-multi-node-boto3
   tensorflow_profiling/low_batch_size
   tensorflow_profiling/dataset_bottleneck
   tensorflow_profiling/callback_bottleneck


PyTorch
-------

.. toctree::
   :maxdepth: 1

   pytorch_profiling/pt-resnet-profiling-single-gpu-single-node.ipynb
   pytorch_profiling/pt-resnet-profiling-multi-gpu-single-node.ipynb
   pytorch_profiling/pt-resnet-profiling-multi-gpu-multi-node
