# Amazon SageMaker Examples

### Prepare Data

The example notebooks within this folder showcase Sagemaker's data preparation capabilities. Data preparation in machine learning refers to the process of collecting, preprocessing, and organizing raw data to make it suitable for analysis and modeling.

- [Data Distribution Types](data_distribution_types) showcases the difference between two methods for sending data from S3 to Amazon SageMaker Training instances.  This has particular implication for scalability and accuracy of distributed training.
- [Distributed Training and Batch Transform with Sentiment Classification](sentiment_parallel_batch) shows how to use SageMaker Distributed Data Parallelism, SageMaker Debugger, and distrubted SageMaker Batch Transform on a HuggingFace Estimator, in a sentiment classification use case.

