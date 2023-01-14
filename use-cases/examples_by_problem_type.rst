SageMaker Algorithms with Pre-Trained Model Examples by Problem Type
====================================================================

The SageMaker Python SDK provides built-in algorithms with pre-trained models from popular open source model hubs, such as TensorFlow Hub, PyTorch Hub, and Hugging Face. Customers can deploy these pre-trained models as-is, or first fine-tune them on a custom dataset and then deploy to a SageMaker endpoint for inference.

This section provides example notebooks for different ML problem types supported by SageMaker built-in algorithms. Please visit `Use Built-in Algorithms with Pre-trained Models in SageMaker Python SDK <https://sagemaker.readthedocs.io/en/stable/overview.html#use-built-in-algorithms-with-pre-trained-models-in-sagemaker-python-sdk>`_ for more documentation.

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
     - .. toctree::
           :maxdepth: 1

           ../introduction_to_amazon_algorithms/jumpstart_object_detection/Amazon_JumpStart_Object_Detection.ipynb
   * - Semantic segmentation
     - Yes
     - Yes
     - MXNet
     - .. toctree::
           :maxdepth: 1

           ../introduction_to_amazon_algorithms/jumpstart_semantic_segmentation/Amazon_JumpStart_Semantic_Segmentation.ipynb
   * - Instance segmentation
     - Yes
     - Yes
     - MXNet
     - .. toctree::
           :maxdepth: 1

           ../introduction_to_amazon_algorithms/jumpstart_instance_segmentation/Amazon_JumpStart_Instance_Segmentation.ipynb
   * - Image embedding
     - Yes
     - No
     - TensorFlow, MXNet
     - .. toctree::
           :maxdepth: 1

           ../introduction_to_amazon_algorithms/jumpstart_image_embedding/Amazon_JumpStart_Image_Embedding.ipynb
   * - Text classification
     - Yes
     - Yes
     - TensorFlow
     - .. toctree::
           :maxdepth: 1

           ../introduction_to_amazon_algorithms/jumpstart_text_classification/Amazon_JumpStart_Text_Classification.ipynb
   * - Sentence pair classification
     - Yes
     - Yes
     - TensorFlow, Hugging Face
     - .. toctree::
           :maxdepth: 1

           ../introduction_to_amazon_algorithms/jumpstart_sentence_pair_classification/Amazon_JumpStart_Sentence_Pair_Classification.ipynb
   * - Question answering
     - Yes
     - Yes
     - PyTorch
     - .. toctree::
           :maxdepth: 1

           ../introduction_to_amazon_algorithms/jumpstart_question_answering/Amazon_JumpStart_Question_Answering.ipynb
   * - Named entity recognition
     - Yes
     - No
     - Hugging Face
     - .. toctree::
           :maxdepth: 1

           ../introduction_to_amazon_algorithms/jumpstart_named_entity_recognition/Amazon_JumpStart_Named_Entity_Recognition.ipynb
   * - Text summarization
     - Yes
     - No
     - Hugging Face
     - .. toctree::
           :maxdepth: 1

           ../introduction_to_amazon_algorithms/jumpstart_text_summarization/Amazon_JumpStart_Text_Summarization.ipynb
   * - Text generation
     - Yes
     - No
     - Hugging Face
     - .. toctree::
           :maxdepth: 1

           ../introduction_to_amazon_algorithms/jumpstart_text_generation/Amazon_JumpStart_Text_Generation.ipynb
   * - Machine translation
     - Yes
     - No
     - Hugging Face
     - .. toctree::
           :maxdepth: 1

           ../introduction_to_amazon_algorithms/jumpstart_machine_translation/Amazon_JumpStart_Machine_Translation.ipynb
   * - Text embedding
     - Yes
     - No
     - TensorFlow, MXNet
     - .. toctree::
           :maxdepth: 1

           ../introduction_to_amazon_algorithms/jumpstart_text_embedding/Amazon_JumpStart_Text_Embedding.ipynb
   * - Tabular classification
     - Yes
     - Yes
     - | LightGBM, CatBoost, XGBoost,
       | AutoGluon-Tabular,
       | TabTransformer, Linear Learner
     - .. toctree::
           :maxdepth: 1

           ../introduction_to_amazon_algorithms/lightgbm_catboost_tabular/Amazon_Tabular_Classification_LightGBM_CatBoost.ipynb
           ../introduction_to_amazon_algorithms/xgboost_linear_learner_tabular/Amazon_Tabular_Classification_XGBoost_LinearLearner.ipynb
           ../introduction_to_amazon_algorithms/autogluon_tabular/Amazon_Tabular_Classification_AutoGluon.ipynb
           ../introduction_to_amazon_algorithms/tabtransformer_tabular/Amazon_Tabular_Classification_TabTransformer.ipynb
   * - Tabular regression
     - Yes
     - Yes
     - | LightGBM, CatBoost, XGBoost,
       | AutoGluon-Tabular,
       | TabTransformer, Linear Learner
     - .. toctree::
           :maxdepth: 1

           ../introduction_to_amazon_algorithms/lightgbm_catboost_tabular/Amazon_Tabular_Classification_LightGBM_CatBoost.ipynb
           ../introduction_to_amazon_algorithms/xgboost_linear_learner_tabular/Amazon_Tabular_Classification_XGBoost_LinearLearner.ipynb
           ../introduction_to_amazon_algorithms/autogluon_tabular/Amazon_Tabular_Classification_AutoGluon.ipynb
           ../introduction_to_amazon_algorithms/tabtransformer_tabular/Amazon_Tabular_Classification_TabTransformer.ipynb
