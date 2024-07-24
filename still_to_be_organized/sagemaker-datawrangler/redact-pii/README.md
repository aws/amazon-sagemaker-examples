# Automatically redact PII for machine learning using Amazon SageMaker Data Wrangler

## Background

Customers increasingly want to use deep learning approaches such as Large Language Models (LLMs) to automate the extraction of data and insights. For many industries, data that is useful for machine learning may contain personally identifiable information (PII). To ensure customer privacy and maintain regulatory compliance while training, fine-tuning, and using deep learning models it is often necessary to first redact PII from source data.

This Data Wrangler flow demonstrates how to use [Amazon SageMaker Data Wrangler](https://aws.amazon.com/sagemaker/data-wrangler/) and [Amazon Comprehend](https://aws.amazon.com/comprehend/) to automatically redact PII from tabular data as part of your machine learning operations (ML Ops) workflow.

## Overview

This solution uses a public [synthetic dataset](https://aws-ml-blog.s3.amazonaws.com/artifacts/fraud-detector-transaction-fraud-insights/synthetic_txn_data_new.csv) along with the custom [Data Wrangler flow]().

The high-level steps to use the Data Wrangler flow to redact PII are:

1. Open SageMaker Studio.
2. Download the Data Wrangler flow.
3. Review the Data Wrangler flow.
4. Add a Destination node.
5. Create an export job.

These steps, including running the export job, should take 20-25 minutes.

## Procedure

For the detailed set of instructions to use the Data Wrangler flow see the [blog post]() associated with the flow.

The blog post also provides a fuller explanation of the dataset, the challenge of PII for machine learning, and how the Data Wrangler flow can fit within your ML Ops workflow.

