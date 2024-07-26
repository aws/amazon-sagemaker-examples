### Scikit-Learn Data Processing and Model Evaluation


This notebook shows how you can:

- run a processing job to run a Scikit-Learn script to clean, pre-process, perform feature engineering, and split the input data into train and test sets.
- run a training job on the pre-processed training data to train a model model
- run a processing job on the pre-processed test data to evaluate the trained model's performance
- use your own custom container with to run processing jobs with your own Python libraries and dependencies.

The dataset used is the [Census-Income KDD Dataset](https://archive.ics.uci.edu/ml/datasets/Census-Income+%28KDD%29). We will select features from this dataset, clean the data, and turn the data into features that our training algorithm can use to train a binary classification model, and split the data into train and test sets.

The task is to predict whether rows representing census responders have an income greater than `$50K`, or less than `50K`. The dataset is heavily class imbalanced, with most records being labeled as earning less than `$50K`. After training a logistic regression model, we will evaluate the model against a hold-out test dataset, and save the classification evaluation metrics, including precision, recall, and F1 score for each label, and accuracy and ROC AUC for the model.

