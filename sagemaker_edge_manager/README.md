# Amazon SageMaker Edge Manager

SageMaker Edge Manager is a new service from Amazon SageMaker that lets you:

+ prepares custom models for edge device hardware
+ includes a runtime for running machine learning inference efficiently on edge devices
+ enables the device to send samples of data from each model securely to SageMaker for relabeling and retraining.

There are two main components to this service:

+ SageMaker Edge Manager in the Cloud
+ SageMaker Edge Agent on the Edge device


These notebooks walks the user through steps for compiling a pre-trained model using AWS SageMaker Neo service. We show how to package this compiled model and then how to use it on devices. In the first notebool we show you how to manually install the agent on the device, load the model and make predictions with. In the second notebook we show you how to use the provided SageMaker EdgeManager Greengrass component for an automatic installation. Finally, we show how to capture model's input and output to S3 via the Agent.

- [SageMaker Edge Example](sagemaker_edge_example/sagemaker_edge_example.ipynb)
- [SageMaker Edge Manager Greengrass Example](sagemaker_edge_example/sagemaker_edge_greengrass_example.ipynb)
