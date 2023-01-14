
.. image:: ../image/data_prep_header.png
  :width: 600
  :alt: data prep


Image data guide
================

An image data pipeline for machine learning is critical for performance during training and inference.
You also need to know the formats and "shapes" of the images that your framework of choice requires.
Additionally, you can further encode images in optimized formats that will speed up your ML processes.
The following guide covers how you can preprocess images using SageMaker's built-in image processing and for PyTorch or TensorFlow training.

The following notebooks will teach you how to download, structure, and preprocess the data before using it to train a model.
We will show you how to perform these tasks with SageMaker Built-in Algorithms, PyTorch, and TensorFlow.


SageMaker Built-in Algorithms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1

   builtin_preprocess_and_train


PyTorch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. toctree::
   :maxdepth: 1

   pytorch_preprocess_and_train


TensorFlow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. toctree::
   :maxdepth: 1

   tensorflow_preprocess_and_train

