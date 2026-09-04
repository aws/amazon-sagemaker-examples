.. image:: _static/sagemaker_gears.jpg
  :width: 600
  :alt: sagemaker_logo

Amazon SageMaker Example Notebooks
==================================

.. image:: https://readthedocs.org/projects/sagemaker-examples-test-website/badge/?version=latest

Welcome to Amazon SageMaker.
This site highlights example Jupyter notebooks for a variety of machine learning use cases that you can run in SageMaker.

This site is based on the `SageMaker Examples repository <https://github.com/aws/amazon-sagemaker-examples>`_ on GitHub.
To run these notebooks, you will need a SageMaker Notebook Instance or SageMaker Studio.
Refer to the SageMaker developer guide's `Get Started <https://docs.aws.amazon.com/sagemaker/latest/dg/gs.html>`_ page to get one of these set up.

On a Notebook Instance, the examples are pre-installed and available from the examples menu item in JupyterLab.
On SageMaker Studio, you will need to open a terminal, go to your home folder, then clone the repo with the following::

   git clone https://github.com/aws/amazon-sagemaker-examples.git

----

.. toctree::
   :maxdepth: 1
   :caption: Introduction

   intro.rst

We recommend the following notebooks as a broad introduction to the capabilities that SageMaker offers. To explore in even more depth, we provide additional notebooks covering even more use cases and frameworks.


.. toctree::
   :maxdepth: 1
   :caption: Prepare data

   prepare_data/sm-marketplace_augmented_ai_with_marketplace_ml_models/sm-marketplace_augmented_ai_with_marketplace_ml_models


.. toctree::
   :maxdepth: 1
   :caption: Build and Train Models

   build_and_train_models/sm-heterogeneous_clusters_for_model_training/sm-heterogeneous_clusters_for_model_training
   build_and_train_models/sm-managed_spot_training_xgboost/sm-managed_spot_training_xgboost
   build_and_train_models/sm-marketplace_build_model_package_for_listing/sm-marketplace_build_model_package_for_listing
   build_and_train_models/sm-regression_xgboost/sm-regression_xgboost
   build_and_train_models/sm-remote_function_pytorch_mnist/sm-remote_function_pytorch_mnist
   build_and_train_models/sm-scikit_build_your_own_container/sm-scikit_build_your_own_container
   build_and_train_models/sm-script_mode_distributed_training_horovod_tensorflow/sm-script_mode_distributed_training_horovod_tensorflow


.. toctree::
   :maxdepth: 1
   :caption: Deploy and Monitor

   deploy_and_monitor/sm-a_b_testing/sm-a_b_testing
   deploy_and_monitor/sm-async_inference_with_python_sdk/sm-async_inference_with_python_sdk
   deploy_and_monitor/sm-deployment_guardrails_update_inference_endpoint_with_linear_traffic_shifting/sm-deployment_guardrails_update_inference_endpoint_with_linear_traffic_shifting
   deploy_and_monitor/sm-mme_with_torchserve/sm-mme_with_torchserve
   deploy_and_monitor/sm-multi_container_endpoint_direct_invocation/sm-multi_container_endpoint_direct_invocation
   deploy_and_monitor/sm-serverless_inference_huggingface_text_classification/sm-serverless_inference_huggingface_text_classification
   deploy_and_monitor/sm-shadow_variant_shadow_api/sm-shadow_variant_shadow_api
   deploy_and_monitor/sm-app_autoscaling_realtime_endpoints_inference_components/sm-app_autoscaling_realtime_endpoints_inference_components


.. toctree::
   :maxdepth: 1
   :caption: ML Ops

   ml_ops/sm-mlflow_pipelines/sm-mlflow_pipelines
