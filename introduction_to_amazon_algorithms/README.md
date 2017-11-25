# Introduction to Amazon Algorithms

This directory includes introductory examples to Amazon SageMaker Algorithms that we have developed so far.  It seeks to provide guidance and examples on basic functionality rather than a detailed scientific review or an implementation on complex, real-world data.

Example Notebooks include:
- *linear_mnist*: Predicts whether a handwritten digit from the MNIST dataset is a 0 or not using a binary classifier from Amazon SageMaker Linear Learner.
- *factorization_machines_mnist*: Predicts whether a handwritten digit from the MNIST dataset is a 0 or not using a binary classifier from Amazon SageMaker Factorization Machines.
- *pca_mnist*: Uses Amazon SageMaker Principal Components Analysis (PCA) to calculate eigendigits from MNIST.
- *ntm_synthetic*: Uses Amazon SageMaker Neural Topic Model (NTM) to uncover topics in documents from a synthetic data source, where topic distributions are known.
- *xgboost_mnist*: Uses Amazon SageMaker XGBoost to classifiy handwritten digits from the MNIST dataset into one of the ten digits using a multi-class classifier. Both single machine and distributed use-cases are presented. 
- *xgboost_abalone*: Predicts the age of abalone ([Abalone dataset](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression.html)) using regression from Amazon SageMaker XGBoost.
