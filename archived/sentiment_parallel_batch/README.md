# Distributed Training and Batch Transform with Sentiment Classification

This example shows how to use [SageMaker Distributed Data Parallelism](https://docs.aws.amazon.com/sagemaker/latest/dg/data-parallel.html), [SageMaker Debugger](https://docs.aws.amazon.com/sagemaker/latest/dg/train-debugger.html), and distrubted [SageMaker Batch Transform](https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform.html) on a [HuggingFace Estimator](https://sagemaker.readthedocs.io/en/stable/frameworks/huggingface/sagemaker.huggingface.html), in a sentiment classification use case. It provides an end-to-end binary text classification example, including the following steps:

- Preprocessing, consisting of downloading the dataset, tokenizing it, and uploading the data to Amazon S3.
- Fine-tuning a HuggingFace Estimator, using SageMaker Data Parallelism across two ml.p3.16xlarge instances. It also uses SageMaker Debugger to save tensors whilst training, and monitor whether the loss is not decreasing. It then prints the saved tensor names, and visualizes the custom metrics captured during the course of the training job.
- Run Batch Transform on the fine-tuned model, using generated dummy data. This Batch Transform step is distributed across two ml.p3.2xlarge instances. It then displays a boxplot which visualizes the ranges and interquartile ranges of the model confidence for positive and negative sentiments.

This example includes:

- [huggingface_sentiment_parallel_batch.ipynb](sentiment_parallel_batch/huggingface_sentiment_parallel_batch.ipynb), the notebook file which walks through the example.
- [train.py](sentiment_parallel_batch/scripts/train.py), the Python script used in the training job.
