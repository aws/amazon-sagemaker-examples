# Ensemble Modeling Example

This example notebook shows how to use mutiple models from SageMaker for prediction and combine then into an ensemble prediction.

It demonstrates the following:
* Basic setup for using SageMaker.
* converting datasets to protobuf format used by the Amazon SageMaker algorithms and uploading to user provided S3 bucket. 
* Training SageMaker's XGBoost algorithm on the data set.
* Training SageMaker's Linear Learner on the data set.
* Hosting the trained models.
* Scoring using the trained models.
* Combining predictions from the trained models in an ensemble.
