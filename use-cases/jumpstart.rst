SageMaker Algorithms with Pre-Trained Model Examples by Problem Type
====================================================================

The SageMaker Python SDK provides built-in algorithms with pre-trained models from popular open source model hubs, such as TensorFlow Hub, PyTorch Hub, and Hugging Face. Customers can deploy these pre-trained models as-is, or first fine-tune them on a custom dataset and then deploy to a SageMaker endpoint for inference.

This section provides example notebooks for different ML problem types supported by SageMaker built-in algorithms. Please visit `Use Built-in Algorithms with Pre-trained Models in SageMaker Python SDK <https://sagemaker.readthedocs.io/en/stable/overview.html#id7>`_ for more documentation.

.. list-table:: Example notebooks by problem type
   :header-rows: 1

   * - | Problem types
     - | Supports
       | inference
       | with
       | pre-trained
       | models
     - | Trainable
       | on a
       | custom
       | dataset
     - | Supported frameworks
     - | Example notebooks
   * - Image classification
     - Yes
     - Yes
     - PyTorch, TensorFlow
     - .. toctree::
           :maxdepth: 1

           ../introduction_to_amazon_algorithms/jumpstart_image_classification/Amazon_JumpStart_Image_Classification
   * - Object detection
     - Yes
     - Yes
     - PyTorch, TensorFlow, MXNet
     -
   * - Semantic segmentation
     - Yes
     - Yes
     - MXNet
     -
   * - Instance segmentation
     - Yes
     - Yes
     - MXNet
     -
   * - Image embedding
     - Yes
     - No
     - TensorFlow, MXNet
     -
   * - Text classification
     - Yes
     - Yes
     - TensorFlow
     -
   * - Sentence pair classification
     - Yes
     - Yes
     - TensorFlow, Hugging Face
     -
   * - Question answering
     - Yes
     - Yes
     - PyTorch
     -
   * - Named entity recognition
     - Yes
     - No
     - Hugging Face
     -
   * - Text summarization
     - Yes
     - No
     - Hugging Face
     -
   * - Text generation
     - Yes
     - No
     - Hugging Face
     -
   * - Machine translation
     - Yes
     - No
     - Hugging Face
     -
   * - Text embedding
     - Yes
     - No
     - TensorFlow, MXNet
     -
   * - Tabular classification
     - Yes
     - Yes
     - | LightGBM, CatBoost, XGBoost,
       | AutoGluon-Tabular,
       | TabTransformer, Linear Learner
     -
   * - Tabular regression
     - Yes
     - Yes
     - | LightGBM, CatBoost, XGBoost,
       | AutoGluon-Tabular,
       | TabTransformer, Linear Learner
     -
