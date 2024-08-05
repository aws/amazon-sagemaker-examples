# Amazon SageMaker Examples

### Build and Train Models

The example notebooks within this folder showcase the capabilities of Amazon SageMaker in building and training machine learning models.

- [Data Distribution Types](data_distribution_types) showcases the difference between two methods for sending data from S3 to Amazon SageMaker Training instances.  This has particular implication for scalability and accuracy of distributed training.
- [Distributed Training and Batch Transform with Sentiment Classification](sentiment_parallel_batch) shows how to use SageMaker Distributed Data Parallelism, SageMaker Debugger, and distrubted SageMaker Batch Transform on a HuggingFace Estimator, in a sentiment classification use case.
