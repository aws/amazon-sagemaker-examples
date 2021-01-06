# Amazon SageMaker Edge Manager

SageMaker Edge Manager is a new service from Amazon SageMaker that lets you:

+ prepares custom models for edge device hardware
+ includes a runtime for running machine learning inference efficiently on edge devices
+ enables the device to send samples of data from each model securely to SageMaker for relabeling and retraining.

There are two main components to this service:

+ SageMaker Edge Manager in the Cloud
+ SageMaker Edge Agent on the Edge device


This notebook walks the user through steps for compiling a pre-trained model using AWS SageMaker Neo service. We show how to package this compiled model and then load it to the Agent on the Edge Device to make predictions with. Finally, we show how to capture model's input and output to S3 via the Agent.

- [SageMaker Edge Example](sagemaker_edge_example)
