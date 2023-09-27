![Amazon SageMaker Data Wrangler](https://github.com/aws/amazon-sagemaker-examples/raw/main/_static/sagemaker-banner.png)

# Amazon SageMaker Data Wrangler Examples

Example flows that demonstrate how to aggregate and prepare data for Machine Learning using Amazon SageMaker Data Wrangler.

## :books: Background

[Amazon SageMaker Data Wrangler](https://aws.amazon.com/sagemaker/data-wrangler/) reduces the time it takes to aggregate and prepare data for ML. From a single interface in SageMaker Studio, you can import data from Amazon S3, Amazon Athena, Amazon Redshift, AWS Lake Formation, and Amazon SageMaker Feature Store, and in just a few clicks SageMaker Data Wrangler will automatically load, aggregate, and display the raw data. It will then make conversion recommendations based on the source data, transform the data into new features, validate the features, and provide visualizations with recommendations on how to remove common sources of error such as incorrect labels. Once your data is prepared, you can build fully automated ML workflows with Amazon SageMaker Pipelines or import that data into Amazon SageMaker Feature Store.



The [SageMaker example notebooks](https://sagemaker-examples.readthedocs.io/en/latest/) are Jupyter notebooks that demonstrate the usage of Amazon SageMaker.

## :hammer_and_wrench: Setup

Amazon SageMaker Data Wrangler is a feature in Amazon SageMaker Studio. Use this section to learn how to access and get started using Data Wrangler. Do the following:

* Complete each step in [Prerequisites](https://docs.aws.amazon.com/sagemaker/latest/dg/data-wrangler-getting-started.html#data-wrangler-getting-started-prerequisite).

* Follow the procedure in [Access Data Wrangler](https://docs.aws.amazon.com/sagemaker/latest/dg/data-wrangler-getting-started.html#data-wrangler-getting-started-access) to start using Data Wrangler.




## :notebook: Examples

### **[Tabular DataFlow](tabular-dataflow/README.md)**

This example provide quick walkthrough of how to aggregate and prepare data for Machine Learning using Amazon SageMaker Data Wrangler for Tabular dataset.

### **[Timeseries DataFlow](timeseries-dataflow/readme.md)**

This example provide quick walkthrough of how to aggregate and prepare data for Machine Learning using Amazon SageMaker Data Wrangler for Timeseries dataset.
### **[Timeseries Quantile Selection DataFlow](timeseries-quantile-selection-dataflow/README.md)**

This example demonstrates how to select quantiles likely to maximize business profitability when using probabilistic time-series forecasting use cases.

### **[Automatically redact PII for machine learning DataFlow](redact-pii/README.md)**

This example provides a walkthrough of how to use Amazon Data Wrangler and Amazon Comprehend to redact personally identifiable information (PII) from tabular datasets.



