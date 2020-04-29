# Amazon SageMaker Examples

### Introduction to Applying Machine Learning

These examples provide a gentle introduction to machine learning concepts as they are applied in practical use cases across a variety of sectors.

- [Targeted Direct Marketing](xgboost_direct_marketing) predicts potential customers that are most likely to convert based on customer and aggregate level metrics, using Amazon SageMaker's implementation of [XGBoost](https://github.com/dmlc/xgboost).
- [Predicting Customer Churn](xgboost_customer_churn) uses customer interaction and service usage data to find those most likely to churn, and then walks through the cost/benefit trade-offs of providing retention incentives.  This uses Amazon SageMaker's implementation of [XGBoost](https://github.com/dmlc/xgboost) to create a highly predictive model.
- [Time-series Forecasting](linear_time_series_forecast) generates a forecast for topline product demand using Amazon SageMaker's Linear Learner algorithm.
- [Cancer Prediction](breast_cancer_prediction) predicts Breast Cancer based on features derived from images, using SageMaker's Linear Learner.
- [Ensembling](ensemble_modeling) predicts income using two Amazon SageMaker models to show the advantages in ensembling.
- [Video Game Sales](video_game_sales) develops a binary prediction model for the success of video games based on review scores.
- [MXNet Gluon Recommender System](gluon_recommender_system) uses neural network embeddings for non-linear matrix factorization to predict user movie ratings on Amazon digital reviews.
- [Fair Linear Learner](fair_linear_learner) is an example of an effective way to create fair linear models with respect to sensitive features.
- [Population Segmentation of US Census Data using PCA and Kmeans](US-census_population_segmentation_PCA_Kmeans) analyzes US census data and reduces dimensionality using PCA then clusters US counties using KMeans to identify segments of similar counties.
- [Document Embedding using Object2Vec](object2vec_document_embedding) is an example to embed a large collection of documents in a common low-dimensional space, so that the semantic distances between these documents are preserved.
