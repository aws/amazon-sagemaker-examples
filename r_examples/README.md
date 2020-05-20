## R on SageMaker Examples

This folder contains examples that are focused on utilizing the R kernel in SageMaker.

[End-2-End Example for using R Kernel in SageMaker](#./r_end_2_end): This sample Notebook describes how to train, deploy, and retrieve predictions from a machine learning (ML) model using Amazon SageMaker and R. The model predicts abalone age as measured by the number of rings in the shell. The reticulate package will be used as an R interface to Amazon SageMaker Python SDK to make API calls to Amazon SageMaker. The reticulate package translates between R and Python objects, and Amazon SageMaker provides a serverless data science environment to train and deploy ML models at scale.

[SageMaker Batch Transform using R Kernel](./r_batch_transform): This sample Notebook describes how to conduct batch transform using SageMaker Transformer in R. The notebook uses Abalone dataset and XGBoost regressor algorithm.

[Bring Your Own R Algorithm to SageMaker for Hyperparamter Optimization](./r_byo_hpo): This notebook will focus mainly on the integration of hyperparameter tuning and a custom algorithm container, as well as hosting the tuned model and making inference using the endpoint.

[Hyperparameter Optimization for XGBoost in R and Batch Transform](./r_xgboost_hpo_batch_transform): This sample Notebook describes how to conduct Hyperparamter tuning and batch transform to make predictions for abalone age as measured by the number of rings in the shell. The notebook will use the public abalone dataset hosted by UCI Machine Learning Repository.

These examples utilize two libraries that provide R interfaces for AWS SageMaker and AWS services:

[`reticulate` library](https://rstudio.github.io/reticulate/): that provides an R interface to make API calls Amazon SageMaker Python SDK to make API calls to Amazon SageMaker. The reticulate package translates between R and Python objects, and Amazon SageMaker provides a serverless data science environment to train and deploy ML models at scale.

[`paws` library](https://cran.r-project.org/web/packages/paws/index.html): that provides an interface to make API calls to AWS services, similar to how boto3 works. boto3 is the Amazon Web Services (AWS) SDK for Python. It enables Python developers to create, configure, and manage AWS services, such as EC2 and S3. Boto provides an easy to use, object-oriented API, as well as low-level access to AWS services. paws provides the same capabilities in R.
