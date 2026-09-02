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

.. toctree::
   :maxdepth: 1
   :caption: Prepare data

   prepare_data/sm-data_wrangler_data_prep_widget/sm-data_wrangler_data_prep_widget
   prepare_data/sm-ground_truth_annotation_consolidation_image_classification/sm-ground_truth_annotation_consolidation_image_classification
   prepare_data/sm-ground_truth_labeling_adjustment_job_adaptation
   prepare_data/sm-ground_truth_rlhf_llm_finetuning
   prepare_data/sm-ground_truth_video_quality_metrics/sm-ground_truth_video_quality_metrics

.. toctree::
   :maxdepth: 1
   :caption: Build and Train Models

   build_and_train_models/sm-deepar_example
   build_and_train_models/sm-forecast_deepar_time_series_modeling/sm-forecast_deepar_time_series_modeling
   build_and_train_models/sm-heterogeneous_clusters_training/sm-heterogeneous_clusters_training
   build_and_train_models/sm-introduction_to_auogluon_tabular_regression
   build_and_train_models/sm-introduction_to_blazingtext_word2vec_text8/sm-introduction_to_blazingtext_word2vec_text8
   build_and_train_models/sm-introduction_to_object2vec_sentence_similarity/sm-introduction_to_object2vec_sentence_similarity
   build_and_train_models/sm-jax_bring_your_own/sm-jax_bring_your_own
   build_and_train_models/sm-jumpstart_private_model_hub_import/sm-jumpstart_private_model_hub_import
   build_and_train_models/sm-lightgbm_catboost_tabular_classification
   build_and_train_models/sm-linear_learner_mnist
   build_and_train_models/sm-marketplace_building_your_own_container_as_package/sm-marketplace_building_your_own_container_as_package
   build_and_train_models/sm-tabtransformer_tabular_classification
   build_and_train_models/sm-training-queues-pytorch/examples/estimator-pytorch-cpu/estimator-pytorch-cpu

.. toctree::
   :maxdepth: 1
   :caption: Deploy and Monitor

   deploy_and_monitor/sm-app_autoscaling_realtime_endpoints/sm-app_autoscaling_realtime_endpoints
   deploy_and_monitor/sm-app_autoscaling_realtime_endpoints_step_scaling/sm-app_autoscaling_realtime_endpoints_step_scaling
   deploy_and_monitor/sm-async_inference_walkthrough/sm-async_inference_walkthrough
   deploy_and_monitor/sm-batch_inference_with_torchserve/sm-batch_inference_with_torchserve
   deploy_and_monitor/sm-deployment_guardrails_update_inference_endpoint_with_rolling_deployment/sm-deployment_guardrails_update_inference_endpoint_with_rolling_deployment
   deploy_and_monitor/sm-deployment_guardrails_update_inference_endpoint_with_with_canary_traffic_shifting/sm-deployment_guardrails_update_inference_endpoint_with_with_canary_traffic_shifting
   deploy_and_monitor/sm-marketplace_using_model_package_arn/sm-marketplace_using_model_package_arn
   deploy_and_monitor/sm-multi_cloud_deployment_with_onnx/pytorch/mnist-train-using-pytorch
   deploy_and_monitor/sm-multi_model_endpoint_bring_your_own_container/sm-multi_model_endpoint_bring_your_own_container

.. toctree::
   :maxdepth: 1
   :caption: Generatative AI

   generative_ai/sm-djl_deepspeed_bloom_176b_deploy

.. toctree::
   :maxdepth: 1
   :caption: SageMaker Core

   sagemaker-core/get_started
   sagemaker-core/inference_and_resource_chaining
   sagemaker-core/intelligent_defaults_and_logging
   sagemaker-core/introductory-blog-sagemaker-core/introductory-blog-sagemaker-core
   sagemaker-core/sagemaker-core-llama-3-8B
   sagemaker-core/sagemaker-core-llama-3-8B-speculative-decoding
   sagemaker-core/sagemaker-core-pyspark-processing
   sagemaker-core/sagemaker_core_overview

