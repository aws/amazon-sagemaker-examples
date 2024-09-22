# Deploying Pre-trained Faster-Whisper-Large-v3 Model on SageMaker with Multi-Model Endpoint and Triton Serve

This guide provides an overview of deploying a pre-trained [Faster-Whisper-Large-v3 model](https://huggingface.co/Systran/faster-whisper-large-v3) from Hugging Face on Amazon SageMaker. The deployment utilizes a Multi-Model Endpoint, allowing efficient serving of multiple models on a single SageMaker instance.

## Overview

1. **Model Selection:** Choose the Faster-Whisper-Large-v3 model from Hugging Face's model hub.
2. **SageMaker Deployment:** Deploy the selected model on Amazon SageMaker, a managed service for ML model training and deployment.
3. **Multi-Model Endpoint:** Utilize a Multi-Model Endpoint on SageMaker for hosting and serving multiple models on a single instance.
4. **Triton Serve Integration:** Integrate Triton Serve, an open-source model serving platform, for dynamic loading and unloading of models from the GPU.

## Cost Optimization

By using a Multi-Model Endpoint on SageMaker, achieve cost savings by maximizing resource utilization and cost efficiency.

## Prerequisites

Ensure [git-lfs](https://git-lfs.com/) is installed.

### Install Dependencies

```bash
%pip install -qU pip awscli boto3 sagemaker
```

### Setting up SageMaker Execution Role

1. Create a SageMaker execution role `AmazonSageMaker-ExecutionRole-mme-FasterWishper` with necessary policies.
2. Assign the created role to your SageMaker instance/job.

## Usage

1. Clone the Faster-Whisper-Large-v3 model from Hugging Face.
2. Arrange the model as Triton server expects it.
3. Create the Triton server configuration file (`config.pbtxt`).
4. Package the model as `*.tar.gz` for uploading to S3.
5. Build and push the Docker image to ECR.
6. Creat SageMaker Multi Model EndPoint base on Triton server
7. Test your endpoint

---

This summarizes the key steps and commands involved in deploying the Faster-Whisper-Large-v3 model on SageMaker with a Multi-Model Endpoint and Triton Serve. Adjustments can be made based on specific requirements and configurations.

Link to the [example notebook](sagemaker-mme-triton-fasterwhisper.ipynb)