# Breast Cancer Prediction with XGBoost
_**Using Gradient Boosted Trees to Predict breast cancer with features derived from breast mass images**_

---
## Background

This notebook illustrates the use of SageMaker's built-in XGBoost algorithm for binary classification.
XGBoost uses decision trees to build a predictive model.

Also demonstrated is Hyperparameter optimization as well as using the best model from HPO to instantiate a new endpoint

### Why XGBoost and not Logistic Regression?

Whilst logistic regression is often used for classification exercises, it has some drawbacks. For example, additional feature engineering is required to deal with non-linear features.

XGBoost (an implementation of Gradient Boosted Trees) offers several benefits including naturally accounting for non-linear relationships between features and the target variable, as well as accommodating complex interactions between features.
Decision Tree algorithms such as XGBoost also have the added benefit of being able to deal with missing values in both the training dataset as well as unseen samples that are being used for inference.

Amazon SageMaker provides an XGBoost container that we can use to train in a managed, distributed setting, and then host as a real-time prediction endpoint.
