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
   :caption: Get started on SageMaker

   introduction_to_applying_machine_learning/xgboost_customer_churn/xgboost_customer_churn_outputs

.. toctree::
   :maxdepth: 1
   :caption: Prepare data

   sagemaker-datawrangler/index
   sagemaker_processing/spark_distributed_data_processing/sagemaker-spark-processing_outputs
   sagemaker_processing/basic_sagemaker_data_processing/basic_sagemaker_processing_outputs



.. toctree::
   :maxdepth: 1
   :caption: Train and tune models

   hyperparameter_tuning/tensorflow2_mnist/hpo_tensorflow2_mnist_outputs
   sagemaker-script-mode/sklearn/sklearn_byom_outputs
   sagemaker-experiments/mnist-handwritten-digits-classification-experiment/mnist-handwritten-digits-classification-experiment_outputs


.. toctree::
   :maxdepth: 1
   :caption: Deploy models

   sagemaker-script-mode/pytorch_bert/deploy_bert_outputs
   sagemaker_neo_compilation_jobs/pytorch_torchvision/pytorch_torchvision_neo_outputs
   sagemaker_batch_transform/pytorch_mnist_batch_transform/pytorch-mnist-batch-transform_outputs


.. toctree::
   :maxdepth: 1
   :caption: Track, monitor, and explain models

   sagemaker-lineage/sagemaker-lineage-multihop-queries_outputs
   sagemaker_model_monitor/introduction/SageMaker-ModelMonitoring_outputs
   sagemaker_processing/fairness_and_explainability/fairness_and_explainability_outputs


.. toctree::
   :maxdepth: 1
   :caption: Orchestrate workflows

   sagemaker-pipelines/tabular/abalone_build_train_deploy/sagemaker-pipelines-preprocess-train-evaluate-batch-transform_outputs
   sagemaker-pipelines/tabular/lambda-step/sagemaker-pipelines-lambda-step_outputs


.. toctree::
   :maxdepth: 1
   :caption: Popular frameworks

   introduction_to_amazon_algorithms/xgboost_abalone/xgboost_abalone_dist_script_mode_outputs
   introduction_to_applying_machine_learning/huggingface_sentiment_classification/huggingface_sentiment_outputs
   sagemaker-python-sdk/scikit_learn_iris/scikit_learn_estimator_example_with_batch_transform_outputs
   sagemaker-python-sdk/mxnet_gluon_mnist/mxnet_mnist_with_gluon_outputs
   frameworks/tensorflow/get_started_mnist_train_outputs
   frameworks/pytorch/get_started_mnist_train_outputs

-----


More examples
=============


.. toctree::
   :maxdepth: 1
   :caption: SageMaker Studio

   aws_sagemaker_studio/index
   sagemaker-lineage/index

.. toctree::
   :maxdepth: 1
   :caption: Introduction to Amazon Algorithms

   introduction_to_amazon_algorithms/index

.. toctree::
   :maxdepth: 1
   :caption: SageMaker End-to-End Examples

   end_to_end/fraud_detection/index
   end_to_end/music_recommendation/index
   end_to_end/nlp_mlops_company_sentiment/index

.. toctree::
    :maxdepth: 1
    :caption: Patterns

    patterns/ml_gateway/index


.. toctree::
    :maxdepth: 1
    :caption: SageMaker Use Cases

    use-cases/index
    use-cases/examples_by_problem_type


.. toctree::
   :maxdepth: 1
   :caption: Autopilot

   autopilot/index


.. toctree::
   :maxdepth: 1
   :caption: Ingest Data

   ingest_data/index


.. toctree::
   :maxdepth: 1
   :caption: Label Data

   label_data/index


.. toctree::
   :maxdepth: 1
   :caption: Prep Data

   prep_data/index


.. toctree::
   :maxdepth: 1
   :caption: Feature Store

   sagemaker-featurestore/index


.. toctree::
   :maxdepth: 1
   :caption: Frameworks

   training/frameworks


.. toctree::
   :maxdepth: 1
   :caption: Training

   training/algorithms
   reinforcement_learning/index
   sagemaker-experiments/index
   sagemaker-debugger/index
   training/tuning
   training/distributed_training/index
   sagemaker-training-compiler/index
   sagemaker-script-mode/index
   training/bring_your_own_container
   training/management
   training/heterogeneous-clusters/index

.. toctree::
   :maxdepth: 1
   :caption: Inference

   inference/index
   model-governance/index

.. toctree::
   :maxdepth: 1
   :caption: Workflows

   sagemaker-pipelines/index
   sagemaker_processing/index
   sagemaker-spark/index
   step-functions-data-science-sdk/index

.. toctree::
   :maxdepth: 1
   :caption: Advanced Functionality

   advanced_functionality/index
   serverless-inference/index

.. toctree::
   :maxdepth: 1
   :caption: Advanced examples

   sagemaker-clarify/index
   scientific_details_of_algorithms/index
   aws_marketplace/index
   
.. toctree::
   :maxdepth: 1
   :caption: Community examples

   contrib/index
