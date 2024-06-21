SageMaker Remote Function
=========================


This folder contains different notebooks and scripts to show how to run your machine learning code as training jobs in SageMaker via remote function feature.

Quick Start
-----------

Follow the quick_start notebook to try out the remote function feature and run sample code as training jobs.

.. toctree::
   :maxdepth: 1

   quick_start/


Train an MNIST model with PyTorch (notebook version)
----------------------------------------------------

See how an existing SageMaker example [here](https://github.com/aws/amazon-sagemaker-examples/blob/07e7b05833863b1e56d966a7ce84d8a373e3fd5d/frameworks/pytorch/get_started_mnist_train.ipynb) is adapted
to use remote function.

.. toctree::
    :maxdepth: 1

    pytorch_mnist_sample_notebook/


Train an MNIST model with PyTorch (script version)
--------------------------------------------------

You can use remote function to run your existing machine learning code in python scripts on SageMaker training. In this example, we convert the Notebook in previous section
to python scripts.

.. toctree::
    :maxdepth: 1

    pytorch_mnist_sample_script/


Text classification using a HuggingFace classifier
--------------------------------------------------

In this example, we use remote function to fine tune a HuggingFace text classifier with IMDB dataset on SageMaker.

.. toctree::
    :maxdepth: 1

    huggingface_text_classification/


Text classification using a HuggingFace classifier
--------------------------------------------------

In this example, we use remote function for regression analysis using XGBoost framework.

.. toctree::
    :maxdepth: 1

    xgboost_abalone/
