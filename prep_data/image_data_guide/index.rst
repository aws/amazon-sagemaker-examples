
.. image:: ../image/data_prep_header.png
  :width: 600
  :alt: data prep


Image data guide
================

An image data pipeline for machine learning is critical for performance during training and inference.
You also need to know the formats and "shapes" of the images that your framework of choice requires.
Additionally, you can further encode images in optimized formats that will speed up your ML processes.
The following guide covers how you can preprocess images using SageMaker's built-in image processing and for PyTorch or TensorFlow training.

To get started, run the following notebooks in order. There are four phases:
   1. Download data
   2. Structure data
   3. Preprocess (choose one of SageMaker built-in, PyTorch, or TensorFlow)
   4. Train (choose one of SageMaker built-in, PyTorch, or TensorFlow)


Download your image data
--------------------------------------
First, download the data.

.. toctree::
   :maxdepth: 1

   01_download_data


Structure your image data
--------------------------------------
Now you structure the data before the next phase which is framework-specific.

.. toctree::
   :maxdepth: 1

   02_structuring_data


Preprocessing
-------------
For preprocessing, you have several options.
This guide covers SageMaker's built-in option and options for PyTorch or TensorFlow.
Choose one of the following notebooks and run it prior to going to the training step for the preprocessing option you chose.

with SageMaker built-in
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1

   03a_builtin_preprocessing


with PyTorch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. toctree::
   :maxdepth: 1

   03c_pytorch_preprocessing


with TensorFlow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. toctree::
   :maxdepth: 1

   03b_tensorflow_preprocessing


Training on image data
----------------------
Now that you preprocessed your image data, choose the corresponding notebook to train with.

with SageMaker built-in
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. toctree::
   :maxdepth: 1

   04a_builtin_training


with PyTorch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. toctree::
   :maxdepth: 1

   04c_pytorch_training


with TensorFlow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. toctree::
   :maxdepth: 1

   04b_tensorflow_training
