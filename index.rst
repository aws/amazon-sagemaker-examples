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
   :caption: End to End ML Lifecycle

   end_to_end_ml_lifecycle/sm-autopilot_customer_churn
   end_to_end_ml_lifecycle/sm-autopilot_linear_regression_california_housing
   end_to_end_ml_lifecycle/sm-autopilot_time_series_forecasting

.. toctree::
   :maxdepth: 1
   :caption: Prepare data

   prepare_data/sm-data_wrangler_data_prep_widget/sm-data_wrangler_data_prep_widget
   prepare_data/sm-feature_store_feature_processor/sm-feature_store_feature_processor
   prepare_data/sm-feature_store_ground_truth_classification_output_to_store/sm-feature_store_ground_truth_classification_output_to_store
   prepare_data/sm-feature_store_introduction/sm-feature_store_introduction
   prepare_data/sm-ground_truth_active_learning_workflow_bring_your_own_model/sm-ground_truth_active_learning_workflow_bring_your_own_model
   prepare_data/sm-ground_truth_annotation_consolidation_image_classification/sm-ground_truth_annotation_consolidation_image_classification
   prepare_data/sm-ground_truth_object_detection_example/sm-ground_truth_object_detection_example
   prepare_data/sm-ground_truth_video_quality_metrics/sm-ground_truth_video_quality_metrics
   prepare_data/sm-marketplace_augmented_ai_with_marketplace_ml_models/sm-marketplace_augmented_ai_with_marketplace_ml_models
   prepare_data/sm-processing_feature_transformation_with_dask/sm-processing_feature_transformation_with_dask
   prepare_data/sm-processing_spark_processing/sm-processing_spark_processing
   prepare_data/sm-spark_pca_kmeans/sm-spark_pca_kmeans
   prepare_data/sm-ground_truth_3d_pointcloud_labeling
   prepare_data/sm-ground_truth_chained_streaming_labeling_job
   prepare_data/sm-ground_truth_create_streaming_labeling_job
   prepare_data/sm-ground_truth_labeling_adjustment_job_adaptation
   prepare_data/sm-ground_truth_object_detection_augmented_manifest_training
   prepare_data/sm-ground_truth_pretrained_model_labeling
   prepare_data/sm-ground_truth_rlhf_llm_finetuning
   prepare_data/sm-ground_truth_text_classification_labeling_accuracy_analysis
   prepare_data/sm-processing_introduction
   prepare_data/sm-processing_r
   prepare_data/sm-spark_kmeans


.. toctree::
   :maxdepth: 1
   :caption: Build and Train Models

   build_and_train_models/sm-distributed_data_parallelism_pytorch/sm-distributed_data_parallelism_pytorch
   build_and_train_models/sm-distributed_model_parallel/sm-distributed_model_parallel
   build_and_train_models/sm-forecast_deepar_time_series_modeling/sm-forecast_deepar_time_series_modeling
   build_and_train_models/sm-fsdp_finetuning_of_llama_v2/sm-fsdp_finetuning_of_llama_v2
   build_and_train_models/sm-heterogeneous_clusters_for_model_training/sm-heterogeneous_clusters_for_model_training
   build_and_train_models/sm-heterogeneous_clusters_training/sm-heterogeneous_clusters_training
   build_and_train_models/sm-hyperparameter_tuning_pytorch/sm-hyperparameter_tuning_pytorch
   build_and_train_models/sm-introduction_to_blazingtext_word2vec_text8/sm-introduction_to_blazingtext_word2vec_text8
   build_and_train_models/sm-introduction_to_ip_insights/sm-introduction_to_ip_insights
   build_and_train_models/sm-introduction_to_lda/sm-introduction_to_lda
   build_and_train_models/sm-introduction_to_ntm/sm-introduction_to_ntm
   build_and_train_models/sm-introduction_to_object2vec_sentence_similarity/sm-introduction_to_object2vec_sentence_similarity
   build_and_train_models/sm-jax_bring_your_own/sm-jax_bring_your_own
   build_and_train_models/sm-managed_spot_training_xgboost/sm-managed_spot_training_xgboost
   build_and_train_models/sm-marketplace_build_model_package_for_listing/sm-marketplace_build_model_package_for_listing
   build_and_train_models/sm-marketplace_building_your_own_container_as_package/sm-marketplace_building_your_own_container_as_package
   build_and_train_models/sm-object_detection_birds/sm-object_detection_birds
   build_and_train_models/sm-random_cut_forest_example/sm-random_cut_forest_example
   build_and_train_models/sm-regression_xgboost/sm-regression_xgboost
   build_and_train_models/sm-remote_function_pytorch_mnist/sm-remote_function_pytorch_mnist
   build_and_train_models/sm-remote_function_quick_start/sm-remote_function_quick_start
   build_and_train_models/sm-scikit_build_your_own_container/sm-scikit_build_your_own_container
   build_and_train_models/sm-script_mode_distributed_training_horovod_tensorflow/sm-script_mode_distributed_training_horovod_tensorflow
   build_and_train_models/sm-semantic_segmentation/sm-semantic_segmentation
   build_and_train_models/sm-smddp_bert/sm-smddp_bert
   build_and_train_models/sm-training_compiler_language_modeling_multi_gpu_multi_node/sm-training_compiler_language_modeling_multi_gpu_multi_node
   build_and_train_models/sm-training_compliler_single-node-single-gpu_bert/sm-training_compliler_single-node-single-gpu_bert
   build_and_train_models/sm-automatic_model_tune_hyperparameter_tuning_early_stopping
   build_and_train_models/sm-deepar_example
   build_and_train_models/sm-distributed_training_model_parallel_v2_mixtral_on_p4
   build_and_train_models/sm-hpo_warmstart_image_classification
   build_and_train_models/sm-hyperparameter_tuning_hyperband_automatic_model_tuning
   build_and_train_models/sm-introduction_to_auogluon_tabular_regression
   build_and_train_models/sm-introduction_to_factorization_machines
   build_and_train_models/sm-introduction_to_pca
   build_and_train_models/sm-k_nearest_neighbors_multi_class_classification
   build_and_train_models/sm-lightgbm_catboost_tabular_classification
   build_and_train_models/sm-linear_learner_mnist
   build_and_train_models/sm-tabtransformer_tabular_classification


.. toctree::
   :maxdepth: 1
   :caption: Deploy and Monitor

   deploy_and_monitor/sm-a_b_testing/sm-a_b_testing
   deploy_and_monitor/sm-async_inference_walkthrough/sm-async_inference_walkthrough
   deploy_and_monitor/sm-async_inference_with_python_sdk/sm-async_inference_with_python_sdk
   deploy_and_monitor/sm-batch_inference_with_torchserve/sm-batch_inference_with_torchserve
   deploy_and_monitor/sm-batch_transform_pca_dbscan_movie_clusters/sm-batch_transform_pca_dbscan_movie_clusters
   deploy_and_monitor/sm-batch_transform_pytorch/sm-batch_transform_pytorch
   deploy_and_monitor/sm-batch_transform_with_torchserve/sm-batch_transform_with_torchserve
   deploy_and_monitor/sm-clarify_model_bias_monitor_batch_transform/sm-clarify_model_bias_monitor_batch_transform
   deploy_and_monitor/sm-clarify_model_bias_monitor_batch_transform_json/sm-clarify_model_bias_monitor_batch_transform_json
   deploy_and_monitor/sm-clarify_model_bias_monitor_for_endpoint/sm-clarify_model_bias_monitor_for_endpoint
   deploy_and_monitor/sm-clarify_model_bias_monitor_for_endpoint_json/sm-clarify_model_bias_monitor_for_endpoint_json
   deploy_and_monitor/sm-deployment_guardrails_update_inference_endpoint_with_linear_traffic_shifting/sm-deployment_guardrails_update_inference_endpoint_with_linear_traffic_shifting
   deploy_and_monitor/sm-deployment_guardrails_update_inference_endpoint_with_rolling_deployment/sm-deployment_guardrails_update_inference_endpoint_with_rolling_deployment
   deploy_and_monitor/sm-deployment_guardrails_update_inference_endpoint_with_with_canary_traffic_shifting/sm-deployment_guardrails_update_inference_endpoint_with_with_canary_traffic_shifting
   deploy_and_monitor/sm-host_pretrained_model_bert/sm-host_pretrained_model_bert
   deploy_and_monitor/sm-inference_pipeline_with_scikit_linear_learner/sm-inference_pipeline_with_scikit_linear_learner
   deploy_and_monitor/sm-lineage_cross_account_queries_with_ram/sm-lineage_cross_account_queries_with_ram
   deploy_and_monitor/sm-marketplace_using_model_package_arn/sm-marketplace_using_model_package_arn
   deploy_and_monitor/sm-mme_with_torchserve/sm-mme_with_torchserve
   deploy_and_monitor/sm-model_monitor_batch_transform_data_quality_on_schedule/sm-model_monitor_batch_transform_data_quality_on_schedule
   deploy_and_monitor/sm-model_monitor_batch_transform_model_quality_on_schedule/sm-model_monitor_batch_transform_model_quality_on_schedule
   deploy_and_monitor/sm-model_monitor_bias_and_explainability_monitoring/sm-model_monitor_bias_and_explainability_monitoring
   deploy_and_monitor/sm-model_monitor_introduction/sm-model_monitor_introduction
   deploy_and_monitor/sm-model_monitor_model_quality_monitoring/sm-model_monitor_model_quality_monitoring
   deploy_and_monitor/sm-multi_container_endpoint_direct_invocation/sm-multi_container_endpoint_direct_invocation
   deploy_and_monitor/sm-multi_model_endpoint_bring_your_own_container/sm-multi_model_endpoint_bring_your_own_container
   deploy_and_monitor/sm-serverless_inference_huggingface_text_classification/sm-serverless_inference_huggingface_text_classification
   deploy_and_monitor/sm-shadow_variant_shadow_api/sm-shadow_variant_shadow_api
   deploy_and_monitor/sm-triton_inferentia2/sm-triton_inferentia2
   deploy_and_monitor/sm-triton_mme_bert_trt/sm-triton_mme_bert_trt
   deploy_and_monitor/sm-triton_mme_gpu_ensemble_dali/sm-triton_mme_gpu_ensemble_dali
   deploy_and_monitor/sm-triton_nlp_bert/sm-triton_nlp_bert
   deploy_and_monitor/sm-triton_realtime_sme_flan_t5/sm-triton_realtime_sme_flan_t5
   deploy_and_monitor/sm-triton_tensorrt-sentence_transformer/sm-triton_tensorrt-sentence_transformer
   deploy_and_monitor/sm-xgboost_bring_your_own_model/sm-xgboost_bring_your_own_model
   deploy_and_monitor/sm-inference_recommender_introduction
   deploy_and_monitor/sm-serverless_inference
   deploy_and_monitor/sm-triton_realtime_sme
   deploy_and_monitor/sm-triton_tensorflow_model_deploy
   deploy_and_monitor/sm-app_autoscaling_realtime_endpoints/sm-app_autoscaling_realtime_endpoints
   deploy_and_monitor/sm-app_autoscaling_realtime_endpoints_inference_components/sm-app_autoscaling_realtime_endpoints_inference_components
   deploy_and_monitor/sm-app_autoscaling_realtime_endpoints_step_scaling/sm-app_autoscaling_realtime_endpoints_step_scaling
   deploy_and_monitor/sm-model_monitor_byoc_llm_monitor/sm-model_monitor_byoc_llm_monitor

.. toctree::
   :maxdepth: 1
   :caption: Generatative AI

   generative_ai/sm-finetuning_huggingface_with_your_own_scripts_and_data/sm-finetuning_huggingface_with_your_own_scripts_and_data
   generative_ai/sm-mixtral_8x7b_fine_tune_and_deploy/sm-mixtral_8x7b_fine_tune_and_deploy
   generative_ai/sm-djl_deepspeed_bloom_176b_deploy
   generative_ai/sm-fsdp_training_of_llama_v2_with_fp8_on_p5
   generative_ai/sm-jumpstart_foundation_code_llama_fine_tuning_human_eval
   generative_ai/sm-jumpstart_foundation_finetuning_gpt_j_6b_domain_adaptation
   generative_ai/sm-jumpstart_foundation_gemma_fine_tuning
   generative_ai/sm-jumpstart_foundation_llama_2_finetuning
   generative_ai/sm-jumpstart_foundation_llama_3_finetuning
   generative_ai/sm-jumpstart_foundation_llama_3_text_completion
   generative_ai/sm-jumpstart_foundation_llama_guard_text_moderation
   generative_ai/sm-jumpstart_foundation_mistral_7b_domain_adaption_finetuning
   generative_ai/sm-jumpstart_foundation_rag_langchain_question_answering
   generative_ai/sm-jumpstart_foundation_text_generation_inference
   generative_ai/sm-jumpstart_foundation_trainium_inferentia_finetuning_deployment
   generative_ai/sm-jumpstart_huggingface_text_classification
   generative_ai/sm-jumpstart_llama_2_chat_completion
   generative_ai/sm-jumpstart_llama_2_text_completion
   generative_ai/sm-jumpstart_rag_question_answering_with_cohere_and_langchain
   generative_ai/sm-jumpstart_stable_diffusion_text_to_image
   generative_ai/sm-jumpstart_text_embedding

.. toctree::
   :maxdepth: 1
   :caption: ML Ops

   ml_ops/sm-ml_lineage_tracking_model_governance_graph/sm-ml_lineage_tracking_model_governance_graph
   ml_ops/sm-mlflow_deployment/sm-mlflow_deployment
   ml_ops/sm-mlflow_hpo/sm-mlflow_hpo
   ml_ops/sm-mlflow_pipelines/sm-mlflow_pipelines
   ml_ops/sm-mlflow_setup/sm-mlflow_setup
   ml_ops/sm-mlflow_training/sm-mlflow_training
   ml_ops/sm-model_monitor_batch_transform_data_quality_with_pipelines_on_demand/sm-model_monitor_batch_transform_data_quality_with_pipelines_on_demand
   ml_ops/sm-model_monitor_batch_transform_model_quality_with_pipelines_on_demand/sm-model_monitor_batch_transform_model_quality_with_pipelines_on_demand
   ml_ops/sm-pipelines_batch_inference_step_decorator/sm-pipelines_batch_inference_step_decorator
   ml_ops/sm-pipelines_callback_step/sm-pipelines_callback_step
   ml_ops/sm-pipelines_clarify_model_monitor_integration/sm-pipelines_clarify_model_monitor_integration
   ml_ops/sm-pipelines_emr_step_using_step_decorator/sm-pipelines_emr_step_using_step_decorator
   ml_ops/sm-pipelines_lambda_step/sm-pipelines_lambda_step
   ml_ops/sm-pipelines_launching_autopilot_with_auto_ml_step/sm-pipelines_launching_autopilot_with_auto_ml_step
   ml_ops/sm-pipelines_local_mode/sm-pipelines_local_mode
   ml_ops/sm-pipelines_preprocess_train_evaluate_batch_transform/sm-pipelines_preprocess_train_evaluate_batch_transform
   ml_ops/sm-pipelines_selective_execution/sm-pipelines_selective_execution
   ml_ops/sm-pipelines_step_caching/sm-pipelines_step_caching
   ml_ops/sm-pipelines_step_decorator/sm-pipelines_step_decorator
   ml_ops/sm-pipelines_step_decorator_with_classic_training_step/sm-pipelines_step_decorator_with_classic_training_step
   ml_ops/sm-pipelines_step_decorator_with_condition_step/sm-pipelines_step_decorator_with_condition_step
   ml_ops/sm-pipelines_emr_step_with_running_emr_cluster
   ml_ops/sm-pipelines_emr-step-with_cluster_lifecycle_management
   ml_ops/sm-pipelines_hyperparameter_tuning
   ml_ops/sm-pipelines_local_mode
   ml_ops/sm-pipelines_train_model_registry_deploy

.. toctree::
   :maxdepth: 1
   :caption: Responsible AI

   responsible_ai/sm-autopilot_model_explanation_with_shap/sm-autopilot_model_explanation_with_shap
   responsible_ai/sm-clarify_explainability_image_classification/sm-clarify_explainability_image_classification
   responsible_ai/sm-clarify_natural_language_processing_online_explainability/sm-clarify_natural_language_processing_online_explainability
   responsible_ai/sm-clarify_object_detection/sm-clarify_object_detection
   responsible_ai/sm-clarify_online_explainability_mme_xgboost/sm-clarify_online_explainability_mme_xgboost
   responsible_ai/sm-clarify_online_explainability_tabular/sm-clarify_online_explainability_tabular
   responsible_ai/sm-clarify_text_explainability_text_sentiment_analysis/sm-clarify_text_explainability_text_sentiment_analysis
   responsible_ai/sm-clarify_time_series_bring_your_own_model/sm-clarify_time_series_bring_your_own_model
   responsible_ai/sm-model_governance_model_card/sm-model_governance_model_card
   responsible_ai/sm-model_governance_model_card_with_model_package/sm-model_governance_model_card_with_model_package
   responsible_ai/sm-clarify_fairness_and_explainability_bring_your_own_container
   responsible_ai/sm-clarify_fairness_and_explainability_with_boto3
