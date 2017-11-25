# Amazon SageMaker Examples

This repository contains example notebooks that show how to apply machine learning and deep learning in Amazon SageMaker(https://aws.amazon.com/amazon-ai/).

## Examples

### Introduction to Applying Machine Learning

- [XGBoost for Direct Marketing](xgboost_direct_marketing) targets potential customers that are most likely to convert based on customer and aggregate level metrics.
- [PCA and k-means for Movie Clustering](pca_kmeans_movie_clustering) creates clusters of movies based on genre, ratings, and other characteristics.

### Introduction to Amazon Algorithms

### Scientific Details of Algorithms

### Advanced Amazon SageMaker Functionality

- [Installing the R Kernel](install_r_kernel) shows how to install the R kernel into an Amazon SageMaker Notebook Instance.
- [Bring Your Own Model for k-means](kmeans_bring_your_own_model) shows how to take a model that's been fit elsewhere and use Amazon SageMaker containers to host.
- [Bring Your Own Algorithm with R](r_bring_your_own) shows how to bring your own algorithm container to Amazon SageMaker using the R language.
- [Bring Your Own Tensorflow Model](sagemaker-python-sdk/tensorflow_iris_byom) shows how to bring a model trained anywhere into Amazon SageMaker

## FAQ

*Will these example work outside of Amazon SageMaker?*

- Although most examples utilize key Amazon SageMaker functionality like distributed, managed training or real-time hosted endpoints, these notebooks can be run outside of Amazon SageMaker Notebook Instances with minimal modification (updating IAM role definition and installing the necessary libraries).

*How do I contribute my own example notebook?"

- Although we're extremely excited to receive contributions from the community, we're still working on the best mechanism to take in examples from and external source.  Please bear will us in the short-term if pull requests take longer than expected or are closed.
