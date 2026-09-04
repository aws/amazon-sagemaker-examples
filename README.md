![SageMaker](https://github.com/aws/amazon-sagemaker-examples/raw/main/_static/sagemaker-banner.png)

# :exclamation::fire: Announcing SageMaker-Core: A New Python SDK for Amazon SageMaker :fire::exclamation:

## Introduction
Today, Amazon SageMaker is excited to announce the release of SageMaker-Core, a new Python SDK that provides an object-oriented interface for interacting with SageMaker resources such as TrainingJob, Model, and Endpoint. This SDK introduces the resource chaining feature, allowing developers to pass resource objects as parameters, eliminating manual parameter specification and simplifying code management. SageMaker-Core abstracts low-level details like resource state transitions and polling logic, achieving full parity with SageMaker APIs. It also includes usability improvements such as auto code completion, comprehensive documentation, and type hints, enhancing the overall developer experience.

## Use Case
SageMaker-Core is ideal for ML practitioners who seek full customization of AWS primitives for their ML workloads. SageMaker-Core is an improvement over Boto3, providing a more intuitive and efficient way to manage SageMaker resources. By providing an intuitive object-oriented interface and resource chaining, the SDK allows for seamless integration and management of SageMaker resources. This flexibility, combined with intelligent defaults enables developers to tailor their ML workloads according to their needs. Comprehensive documentation, and type hints help developers write code faster and with fewer errors without navigating complex API documentation.

## Call to Action
To learn more about SageMaker-Core, visit the [documentation](https://sagemaker-core.readthedocs.io) and [example notebooks](https://github.com/aws/amazon-sagemaker-examples/tree/default). Get started today by integrating SageMaker-Core into your machine learning workflows and experience the benefits of a streamlined and efficient development process.


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

The notebooks are organized by ML capability, following the lifecycle order of a typical
project: train a model, customize it, evaluate it, deploy it, then operationalize it.
Every notebook is self-contained: it lives with the scripts, data and images it needs, and
all of its references are relative to its own folder.

### [Training](https://github.com/aws/amazon-sagemaker-examples/tree/default/%20%20%20%20%20%20training)

Amazon SageMaker Training is a fully managed service that helps you train ML models at
scale. It containerizes your workload and manages the AWS compute for you, so you can focus
on the model rather than the infrastructure. These examples train with `ModelTrainer` from
SageMaker Python SDK v3 and cover script and framework training, distributed training,
managed spot training with checkpointing, heterogeneous clusters, bringing your own
container, submitting work through AWS Batch training queues, and running local code as a
training job with `@remote`.

### [Model Customization](https://github.com/aws/amazon-sagemaker-examples/tree/default/%20%20%20%20%20model_customization)

Model customization adapts a pre-trained foundation model to your data and your task, which
is usually far cheaper and faster than training from scratch. These examples cover the
fine-tuning techniques SageMaker supports — supervised fine-tuning (SFT), direct preference
optimization (DPO), reinforcement learning with verifiable rewards (RLVR), RL from AI
feedback (RLAIF), multi-turn RL (MTRL) and continued pre-training (CPT) — along with recipe
overrides, data mixing, JumpStart fine-tuning with a private model hub, and distributed
fine-tuning across serverless, serverful and HyperPod compute.

### [Evaluation](https://github.com/aws/amazon-sagemaker-examples/tree/default/%20%20%20%20evaluation)

Before you promote a customized model you need to know whether it actually got better.
These examples use the SageMaker evaluator surface to score models: standard benchmark
evaluation, your own scoring function, an LLM acting as a judge, and Inspect AI. They show
how to evaluate a base model and a fine-tuned model on the same footing so the comparison
is meaningful.

### [Inference](https://github.com/aws/amazon-sagemaker-examples/tree/default/%20%20%20inference)

With Amazon SageMaker you can get predictions from your trained models through a broad set
of deployment options, and scale them without taking on the operational burden yourself.
These examples deploy with `ModelBuilder` and sagemaker-core, and cover real-time,
serverless and asynchronous endpoints; in-process and local-container modes for fast
iteration; inference pipelines; multi-model and multi-container endpoints; safe rollout
through A/B testing, shadow variants and deployment guardrails; autoscaling with inference
components; latency and cost optimization with `optimize()`; batch transform; deployment to
Amazon Bedrock; AWS Marketplace model packages; and human review with Augmented AI.

### [MLOps](https://github.com/aws/amazon-sagemaker-examples/tree/default/%20%20mlops)

MLOps applies DevOps practices to machine learning: continuous integration and deployment,
reproducibility, lineage and governance. It accounts for the parts of an AI/ML project that
ordinary software delivery does not, which shortens delivery time, reduces defects and makes
data science more productive. These examples cover SageMaker Pipelines with the Model
Registry, lineage tracking, experiment tracking, Feature Store, processing jobs, EMR
Serverless steps, bias and explainability analysis with Clarify, and MLflow.

## :balance_scale: License

This library is licensed under the [Apache 2.0 License](http://aws.amazon.com/apache2.0/).
For more details, please take a look at the [LICENSE](https://github.com/aws/amazon-sagemaker-examples/blob/master/LICENSE.txt) file.

## :handshake: Contributing

Although we're extremely excited to receive contributions from the community, we're still working on the best mechanism to take in examples from external sources. Please bear with us in the short-term if pull requests take longer than expected or are closed.
Please read our [contributing guidelines](https://github.com/aws/amazon-sagemaker-examples/blob/default/CONTRIBUTING.md)
if you'd like to open an issue or submit a pull request.
