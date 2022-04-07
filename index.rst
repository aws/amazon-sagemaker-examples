.. image:: _static/sagemaker_gears.jpg
  :width: 600
  :alt: sagemaker_logo

Amazon SageMaker Example Notebooks
==================================

.. image:: https://readthedocs.org/projects/sagemaker-examples-test-website/badge/?version=latest

Welcome to Amazon SageMaker.
This site highlights example Jupyter notebooks for a variety of machine learning use cases that you can run in SageMaker.

This site is based on the `SageMaker Examples repository <https://github.com/aws/amazon-sagemaker-examples>`_ on GitHub.
Browse around to see what piques your interest.
To run these notebooks, you will need a SageMaker Notebook Instance or SageMaker Studio.
Refer to the SageMaker developer guide's `Get Started <https://docs.aws.amazon.com/sagemaker/latest/dg/gs.html>`_ page to get one of these set up.

On a Notebook Instance, the examples are pre-installed and available from the examples menu item in JupyterLab.
On SageMaker Studio, you will need to open a terminal, go to your home folder, then clone the repo with the following::

   git clone https://github.com/aws/amazon-sagemaker-examples.git

Get started on SageMaker
========================

.. toctree::
   :maxdepth: 1

   introduction_to_applying_machine_learning/xgboost_customer_churn/xgboost_customer_churn_outputs


Try machine learning on SageMaker
=================================

Prepare data
------------

.. toctree::
   :maxdepth: 1

   sagemaker_processing/basic_sagemaker_data_processing/basic_sagemaker_processing_outputs


Train and tune models
---------------------

.. toctree::
   :maxdepth: 1

   hyperparameter_tuning/tensorflow2_mnist/hpo_tensorflow2_mnist_outputs
   sagemaker-script-mode/sklearn/sklearn_byom_outputs
   sagemaker-experiments/mnist-handwritten-digits-classification-experiment/mnist-handwritten-digits-classification-experiment_outputs


Deploy models
-------------

.. toctree::
   :maxdepth: 1

   sagemaker-script-mode/pytorch_bert/deploy_bert_outputs
   sagemaker_neo_compilation_jobs/pytorch_torchvision/pytorch_torchvision_neo_outputs
   sagemaker_batch_transform/pytorch_mnist_batch_transform/pytorch-mnist-batch-transform_outputs


Track, monitor, and explain models
----------------------------------

.. toctree::
   :maxdepth: 1

   sagemaker-lineage/sagemaker-lineage-multihop-queries_outputs
   sagemaker_model_monitor/introduction/SageMaker-ModelMonitoring_outputs
   sagemaker_processing/fairness_and_explainability/fairness_and_explainability_outputs


Orchestrate workflows
---------------------

.. toctree::
   :maxdepth: 1

   sagemaker-pipelines/tabular/abalone_build_train_deploy/sagemaker-pipelines-preprocess-train-evaluate-batch-transform_outputs
   sagemaker-pipelines/tabular/lambda-step/sagemaker-pipelines-lambda-step_outputs


Popular frameworks
==================

XGBoost
-------

.. toctree::
   :maxdepth: 1

   introduction_to_amazon_algorithms/xgboost_abalone/xgboost_abalone_dist_script_mode_outputs


Hugging Face
------------

.. toctree::
   :maxdepth: 1

   introduction_to_applying_machine_learning/huggingface_sentiment_classification/huggingface_sentiment_outputs


Scikit-Learn
------------

.. toctree::
   :maxdepth: 1

   sagemaker-python-sdk/scikit_learn_iris/scikit_learn_estimator_example_with_batch_transform_outputs


MXNet
-----

.. toctree::
   :maxdepth: 1

   sagemaker-python-sdk/mxnet_gluon_mnist/mxnet_mnist_with_gluon_outputs


TensorFlow
----------

.. toctree::
   :maxdepth: 1
   advanced_functionality/tensorflow_iris_byom/tensorflow_BYOM_iris_outputs


PyTorch
-------

.. toctree::
   :maxdepth: 1

   frameworks/pytorch/get_started_mnist_train_outputs


Advanced examples
=================

.. toctree::
   :maxdepth: 1
   :caption: Get started

   get_started/index


.. toctree::
   :maxdepth: 1
   :caption: SageMaker Studio

   aws_sagemaker_studio/index
   sagemaker-lineage/index


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


.. toctree::
   :maxdepth: 1
   :caption: Inference

   inference/index


.. toctree::
   :maxdepth: 1
   :caption: Workflows

   sagemaker-pipelines/index
   sagemaker_processing/index
   sagemaker-spark/index
   step-functions-data-science-sdk/index


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
