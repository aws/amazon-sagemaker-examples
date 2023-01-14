#########################
Algorithms
#########################

Examples on how to use SageMaker's built-in algorithms.


Image processing
====================================

SageMaker provides algorithms that are used for image processing.

image_classification
--------------------

.. toctree::
   :maxdepth: 1

   ../introduction_to_amazon_algorithms/imageclassification_caltech/Image-classification-fulltraining-elastic-inference
   ../introduction_to_amazon_algorithms/imageclassification_caltech/Image-classification-lst-format
   ../introduction_to_amazon_algorithms/imageclassification_caltech/Image-classification-incremental-training-highlevel
   ../introduction_to_amazon_algorithms/imageclassification_caltech/Image-classification-lst-format-highlevel
   ../introduction_to_amazon_algorithms/imageclassification_caltech/Image-classification-transfer-learning
   ../introduction_to_amazon_algorithms/imageclassification_caltech/Image-classification-transfer-learning-highlevel
   ../introduction_to_amazon_algorithms/imageclassification_caltech/Image-classification-fulltraining
   ../introduction_to_amazon_algorithms/imageclassification_caltech/Image-classification-fulltraining-highlevel
   ../introduction_to_amazon_algorithms/imageclassification_mscoco_multi_label/Image-classification-multilabel-lst


object_detection
----------------

.. toctree::
   :maxdepth: 1

   ../introduction_to_amazon_algorithms/object_detection_birds/object_detection_birds


semantic_segmentation
---------------------

.. toctree::
   :maxdepth: 1

   ../introduction_to_amazon_algorithms/semantic_segmentation_pascalvoc/semantic_segmentation_pascalvoc


Text processing
====================================

SageMaker provides algorithms that are tailored to the analysis of texts and documents used in natural language processing and translation.

blazingtext
-----------

.. toctree::
   :maxdepth: 1

   ../introduction_to_amazon_algorithms/blazingtext_text_classification_dbpedia/blazingtext_text_classification_dbpedia
   ../introduction_to_amazon_algorithms/blazingtext_word2vec_subwords_text8/blazingtext_word2vec_subwords_text8
   ../introduction_to_amazon_algorithms/blazingtext_word2vec_text8/blazingtext_word2vec_text8


lda
-----

.. toctree::
   :maxdepth: 1

   ../introduction_to_amazon_algorithms/lda_topic_modeling/LDA-Introduction


ntm
-----

.. toctree::
   :maxdepth: 1

   ../scientific_details_of_algorithms/ntm_topic_modeling/ntm_wikitext
   ../introduction_to_amazon_algorithms/ntm_synthetic/ntm_synthetic
   ../introduction_to_applying_machine_learning/ntm_20newsgroups_topic_modeling/ntm_20newsgroups_topic_model


seq2seq
-------

.. toctree::
   :maxdepth: 1

   ../introduction_to_amazon_algorithms/seq2seq_translation_en-de/SageMaker-Seq2Seq-Translation-English-German


Time series processing
======================

SageMaker DeepAR algorithm is useful for processing time series data.

deepar
------

.. toctree::
   :maxdepth: 1

   ../introduction_to_amazon_algorithms/deepar_synthetic/deepar_synthetic
   ../introduction_to_amazon_algorithms/deepar_electricity/DeepAR-Electricity
   ../introduction_to_applying_machine_learning/deepar_chicago_traffic_violations/deepar_chicago_traffic_violations

Supervised learning algorithms
====================================

Amazon SageMaker provides several built-in general purpose algorithms that can be used for either classification or regression problems.


factorization_machines
----------------------

.. toctree::
   :maxdepth: 1

   ../introduction_to_amazon_algorithms/factorization_machines_mnist/factorization_machines_mnist


knn
---

.. toctree::
   :maxdepth: 1

   ../introduction_to_amazon_algorithms/k_nearest_neighbors_covtype/k_nearest_neighbors_covtype


linear_learner
--------------

.. toctree::
   :maxdepth: 1

   ../introduction_to_amazon_algorithms/linear_learner_mnist/linear_learner_mnist
   ../introduction_to_amazon_algorithms/linear_learner_mnist/linear_learner_mnist_with_file_system_data_source
   ../scientific_details_of_algorithms/linear_learner_multiclass_classification/linear_learner_multiclass_classification
   ../introduction_to_applying_machine_learning/fair_linear_learner/fair_linear_learner




xgboost
-------

Basic
^^^^^

.. toctree::
   :maxdepth: 1

   ../introduction_to_amazon_algorithms/xgboost_mnist/xgboost_mnist
   ../introduction_to_amazon_algorithms/xgboost_abalone/xgboost_abalone
   ../introduction_to_applying_machine_learning/xgboost_customer_churn/xgboost_customer_churn


Advanced
^^^^^^^^

.. toctree::
   :maxdepth: 1

   ../introduction_to_amazon_algorithms/xgboost_abalone/xgboost_abalone_dist_script_mode
   ../introduction_to_amazon_algorithms/xgboost_abalone/xgboost_parquet_input_training


Unsupervised learning algorithms
====================================

Amazon SageMaker provides several built-in algorithms that can be used for a variety of unsupervised learning tasks such as clustering, dimension reduction, pattern recognition, and anomaly detection.


ip_insights
------------

.. toctree::
   :maxdepth: 1

   ../introduction_to_amazon_algorithms/ipinsights_login/ipinsights-tutorial


kmeans
------

.. toctree::
   :maxdepth: 1

   ../introduction_to_applying_machine_learning/US-census_population_segmentation_PCA_Kmeans/sagemaker-countycensusclustering
   ../sagemaker-python-sdk/1P_kmeans_highlevel/kmeans_mnist
   ../sagemaker-python-sdk/1P_kmeans_lowlevel/kmeans_mnist_lowlevel


pca
-----

.. toctree::
   :maxdepth: 1

   ../introduction_to_amazon_algorithms/pca_mnist/pca_mnist


rcf
-----

.. toctree::
   :maxdepth: 1

   ../introduction_to_amazon_algorithms/random_cut_forest/random_cut_forest


Feature engineering
===================

object2vec
----------

.. toctree::
   :maxdepth: 1

   ../introduction_to_applying_machine_learning/object2vec_document_embedding/object2vec_document_embedding
   ../introduction_to_amazon_algorithms/object2vec_movie_recommendation/object2vec_movie_recommendation
   ../introduction_to_amazon_algorithms/object2vec_sentence_similarity/object2vec_sentence_similarity
