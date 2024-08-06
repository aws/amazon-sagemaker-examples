![SageMaker](https://github.com/aws/amazon-sagemaker-examples/raw/main/_static/sagemaker-banner.png)

# :exclamation::fire: Announcement: New Folder Structure :fire::exclamation:

Starting 7 August, 2024, we are introducing a new flattened folder structure within the SageMaker Example Notebooks repository in order to improve the discoverability of the notebooks. The new structure uses standardized folder and notebook naming conventions to align with common workflows and SageMaker services. It also includes an archived folder for redundant, outdated, and low-viewed notebooks. To ease the transition, we have created an [excel sheet](new_file_structure_updated_notebook_names_and_folders.xlsx) showing the old naming convention and its new corresponding name.

# Amazon SageMaker Examples

Example Jupyter notebooks that demonstrate how to build, train, and deploy machine learning models using Amazon SageMaker.

## :books: Read this before you proceed further

Amazon SageMaker examples are divided in two repositories:

- [SageMaker example notebooks](https://github.com/aws/amazon-sagemaker-examples) is the official repository, containing examples that demonstrate the usage of Amazon SageMaker. This repository is entirely focussed on covering the breadth of features provided by SageMaker, and is maintained directly by the Amazon SageMaker team.

- [Sagemaker Example Community repository](https://github.com/aws/amazon-sagemaker-examples-community) is another SageMaker repository which contains additional examples and reference solutions, beyond the examples showcased in the [official repository](https://github.com/aws/amazon-sagemaker-examples). This repository is maintained by community of engineers and solution architects at AWS.

## Planning to submit a PR to this repository? Read this first:

- This repository will only accept notebooks/examples which demonstrate a feature of SageMaker, not yet covered anywhere in this repository. PR submitters are requested to check this before submitting the PR to avoid getting it rejected.

- If you still would like to contribute your example, please submit a PR to [Sagemaker Example Community repository](https://github.com/aws/amazon-sagemaker-examples-community) instead.

## :hammer_and_wrench: Setup

The quickest setup to run example notebooks includes:

- An [AWS account](http://docs.aws.amazon.com/sagemaker/latest/dg/gs-account.html)
- Proper [IAM User and Role](http://docs.aws.amazon.com/sagemaker/latest/dg/authentication-and-access-control.html) setup
- An [Amazon SageMaker Notebook Instance](http://docs.aws.amazon.com/sagemaker/latest/dg/gs-setup-working-env.html)
- An [S3 bucket](http://docs.aws.amazon.com/sagemaker/latest/dg/gs-config-permissions.html)

## :computer: Usage

These example notebooks are automatically loaded into SageMaker Notebook Instances.
They can be accessed by clicking on the `SageMaker Examples` tab in Jupyter or the SageMaker logo in JupyterLab.

Although most examples utilize key Amazon SageMaker functionality like distributed, managed training or real-time hosted endpoints, these notebooks can be run outside of Amazon SageMaker Notebook Instances with minimal modification (updating IAM role definition and installing the necessary libraries).

## :notebook: Example Notebook Categories

### End-to-End ML Lifecycle

These examples are a diverse collection of end-to-end notebooks that demonstrate how to build, train, and deploy machine learning models using Amazon SageMaker. These notebooks cover a wide range of machine learning tasks and use cases, providing you with a comprehensive understanding of the SageMaker workflow. Each notebook in this folder is self-contained and includes detailed documentation, code samples, and instructions for running the examples on SageMaker. Whether you're a beginner or an experienced practitioner, this folder offers a comprehensive collection of end-to-end notebooks that will help you leverage the power of Amazon SageMaker for a wide range of machine learning tasks and use cases.

### Prepare Data

The example notebooks within this folder showcase Sagemaker's data preparation capabilities. Data preparation in machine learning refers to the process of collecting, preprocessing, and organizing raw data to make it suitable for analysis and modeling. This step ensures that the data is in a format from which machine learning algorithms can effectively learn. Data preparation tasks may include handling missing values, removing outliers, scaling features, encoding categorical variables, assessing potential biases and taking steps to mitigate them, splitting data into training and testing sets, labeling, and other necessary transformations to optimize the quality and usability of the data for subsequent machine learning tasks.

### Build and Train Models

Amazon SageMaker Training is a fully managed machine learning (ML) service offered by SageMaker that helps you efficiently build and train a wide range of ML models at scale. The core of SageMaker jobs is the containerization of ML workloads and the capability of managing AWS compute resources. The SageMaker Training platform takes care of the heavy lifting associated with setting up and managing infrastructure for ML training workloads. With SageMaker Training, you can focus on building, developing, training, and fine-tuning your model.

### Deploy and Monitor

With Amazon SageMaker, you can start getting predictions, or inferences, from your trained machine learning models. SageMaker provides a broad selection of ML infrastructure and model deployment options to help meet all your ML inference needs. With SageMaker Inference, you can scale your model deployment, manage models more effectively in production, and reduce operational burden. SageMaker provides you with various inference options, such as real-time endpoints for getting low latency inference, serverless endpoints for fully managed infrastructure and auto-scaling, and asynchronous endpoints for batches of requests. By leveraging the appropriate inference option for your use case, you can ensure efficient and model deployment and inference.

After you deploy a model into your production environment, use Amazon SageMaker model monitor to continuously monitor the quality of your machine learning models in real time. Amazon SageMaker model monitor enables you to set up an automated alert triggering system when there are deviations in the model quality, such as data drift and anomalies. Amazon CloudWatch Logs collects log files of monitoring the model status and notifies when the quality of your model hits certain thresholds that you preset. CloudWatch stores the log files to an Amazon S3 bucket you specify. Early and pro-active detection of model deviations through AWS model monitor products enables you to take prompt actions to maintain and improve the quality of your deployed model.

### Generative AI

These examples showcases Amazon SageMaker's capabilities in the exciting field of generative artificial intelligence (AI). Generative AI models are designed to create new, synthetic data across various modalities, such as text, images, audio, and video, based on the patterns and relationships learned from training data. These examples provide detailed documentation, code samples, and instructions for running the generative AI models on SageMaker. And demonstrate how to preprocess data, train models, fine-tune hyperparameters, and deploy the trained models for inference.

Whether you're interested in exploring the latest advancements in generative AI, or seeking to leverage these techniques for creative applications or content generation, this folder offers a comprehensive collection of examples that will help you unlock the power of SageMaker's generative AI capabilities and push the boundaries of what's possible with machine learning.

### ML Ops

Amazon SageMaker supports features to implement machine learning models in production environments with continuous integration and deployment. MLOps accounts for the unique aspects of AI/ML projects in project management, CI/CD, and quality assurance, helping you improve delivery time, reduce defects, and make data science more productive. MLOps refers to a methodology that is built on applying DevOps practices to machine learning workloads.

### Responsible AI

Amazon SageMaker offers features to improve your machine learning (ML) models by detecting potential bias and helping to explain the predictions that your models make from your tabular, computer vision, natural processing, or time series datasets. It helps you identify various types of bias in pre-training data and in post-training that can emerge during model training or when the model is in production. You can also evaluate a language model for model quality and responsibility metrics using foundation model evaluations.

Model governance is a framework that gives systematic visibility into machine learning (ML) model development, validation, and usage. Amazon SageMaker provides purpose-built ML governance tools for managing control access, activity tracking, and reporting across the ML lifecycle. Manage least-privilege permissions for ML practitioners using Amazon SageMaker Role Manager, create detailed model documentation using Amazon SageMaker Model Cards, and gain visibility into your models with centralized dashboards using Amazon SageMaker Model Dashboard.

## :balance_scale: License

This library is licensed under the [Apache 2.0 License](http://aws.amazon.com/apache2.0/).
For more details, please take a look at the [LICENSE](https://github.com/aws/amazon-sagemaker-examples/blob/master/LICENSE.txt) file.

## :handshake: Contributing

Although we're extremely excited to receive contributions from the community, we're still working on the best mechanism to take in examples from external sources. Please bear with us in the short-term if pull requests take longer than expected or are closed.
Please read our [contributing guidelines](https://github.com/aws/amazon-sagemaker-examples/blob/default/CONTRIBUTING.md)
if you'd like to open an issue or submit a pull request.
