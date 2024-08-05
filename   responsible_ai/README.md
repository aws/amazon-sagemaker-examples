# Amazon SageMaker Examples

### Responsible AI

Amazon SageMaker offers features to improve your machine learning (ML) models by detecting potential bias and helping to explain the predictions that your models make from your tabular, computer vision, natural processing, or time series datasets as well as providing purpose-built ML governance tools for managing control access, activity tracking, and reporting across the ML lifecycle.

- [Data Distribution Types](data_distribution_types) showcases the difference between two methods for sending data from S3 to Amazon SageMaker Training instances.  This has particular implication for scalability and accuracy of distributed training.
- [Distributed Training and Batch Transform with Sentiment Classification](sentiment_parallel_batch) shows how to use SageMaker Distributed Data Parallelism, SageMaker Debugger, and distrubted SageMaker Batch Transform on a HuggingFace Estimator, in a sentiment classification use case.
