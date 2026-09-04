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
   :caption: Training

         training/distributed-local-training-example
         training/jumpstart-e2e-training-example
         training/sm-heterogeneous_clusters_for_model_training/sm-heterogeneous_clusters_for_model_training
         training/sm-managed_spot_training_xgboost/sm-managed_spot_training_xgboost
         training/sm-regression_xgboost/sm-regression_xgboost
         training/sm-remote_function_pytorch_mnist/sm-remote_function_pytorch_mnist
         training/sm-scikit_build_your_own_container/sm-scikit_build_your_own_container
         training/sm-script_mode_distributed_training_horovod_tensorflow/sm-script_mode_distributed_training_horovod_tensorflow
         training/sm-training-queues_getting_started_with_model_trainer/sm-training-queues_getting_started_with_model_trainer

.. toctree::
   :maxdepth: 1
   :caption: Model Customization

        model_customization/cpt_data_mixing_hyperpod
        model_customization/custom-distributed-training-example
        model_customization/dpo_trainer_example_notebook_v3_prod
        model_customization/estimator-pytorch-gpu/estimator-pytorch-gpu
        model_customization/mtrl_finetuning_example_notebook_v3_prod
        model_customization/nova_data_mixing
        model_customization/recipe_override_model_trainer_example
        model_customization/recipe_override_sft_trainer_example
        model_customization/rlaif_finetuning_example_notebook_v3_prod
        model_customization/rlvr_finetuning_example_notebook_v3_prod
        model_customization/serverless_e2e_example
        model_customization/sft_finetuning_example_notebook_pysdk_prod_v3
        model_customization/sft_finetuning_hyperpod
        model_customization/sft_finetuning_serverful_smtj
        model_customization/sm-jumpstart_private_model_hub_import_llama3-8B/sm-jumpstart_private_model_hub_import_llama3-8B

.. toctree::
   :maxdepth: 1
   :caption: Evaluation

       evaluation/benchmark_demo
       evaluation/custom_scorer_demo
       evaluation/inspect_ai_evaluation_demo
       evaluation/llm_as_judge_demo

.. toctree::
   :maxdepth: 1
   :caption: Inference

      inference/bedrock-modelbuilder-deployment
      inference/in-process-mode-example
      inference/inference-pipeline-modelbuilder-vs-core-example
      inference/local-mode-example
      inference/optimize-example
      inference/sm-a_b_testing/sm-a_b_testing
      inference/sm-app_autoscaling_realtime_endpoints_inference_components/sm-app_autoscaling_realtime_endpoints_inference_components
      inference/sm-async_inference_with_python_sdk/sm-async_inference_with_python_sdk
      inference/sm-deployment_guardrails_update_inference_endpoint_with_linear_traffic_shifting/sm-deployment_guardrails_update_inference_endpoint_with_linear_traffic_shifting
      inference/sm-marketplace_augmented_ai_with_marketplace_ml_models/sm-marketplace_augmented_ai_with_marketplace_ml_models
      inference/sm-marketplace_build_model_package_for_listing/sm-marketplace_build_model_package_for_listing
      inference/sm-mme_with_torchserve/sm-mme_with_torchserve
      inference/sm-multi_container_endpoint_direct_invocation/sm-multi_container_endpoint_direct_invocation
      inference/sm-serverless_inference_huggingface_text_classification/sm-serverless_inference_huggingface_text_classification
      inference/sm-shadow_variant_shadow_api/sm-shadow_variant_shadow_api

.. toctree::
   :maxdepth: 1
   :caption: MLOps

     mlops/sagemaker_core_feature_store_introduction/sagemaker_core_feature_store_introduction
     mlops/sm-mlflow_pipelines/sm-mlflow_pipelines
     mlops/track_local_pytorch_experiment/track_local_pytorch_experiment
     mlops/v3-emr-serverless-step-example
     mlops/v3-lineage-tracking-example
     mlops/v3-pipeline-train-create-registry
     mlops/v3-pytorch-processing-example/v3-pytorch-processing-example
     mlops/v3-sagemaker-clarify
