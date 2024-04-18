# Pre-train Llama2 70B on Redpajama dataset using Neuronx-Nemo-Megatron

This example shows how to pre-train Llama2-70B model on [Redpajama dataset](https://github.com/togethercomputer/RedPajama-Data) with [Neuronx-Nemo-Megatron](https://github.com/aws-neuron/neuronx-nemo-megatron). 

## Prerequisites

Before proceeding, please fulfil the prerequisites shown below.

### Download Hugging Face Llama2 70B pre-trained model weights

You must execute this step on a machine where you have [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) and [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli) installed. Verify you have sufficient disk space on the machine to download the pre-trained model weights.

Below, we show sample commands for downloading and uploading the pre-trained model weights for Hugging Face Llama2 70B model. Replace HF_TOKEN with your Hugging Face CLI token, replace S3_BUCKET with `S3BucketName` and S3_PREFIX with `FSxS3ImportPrefix` values, respectively, used with the [CloudFormation template](../../../cfn-sagemaker-notebook.yaml):

    huggingface-cli download --repo-type model --revision 97a63a0ef386379ab4175785da8352fbe338c6c5 --local-dir ./meta-llama/Llama-2-70b-hf --token HF_TOKEN  meta-llama/Llama-2-70b-hf

    aws s3 cp --recursive meta-llama/Llama-2-70b-hf s3://S3_BUCKET/ml-platform/pretrained-models/meta-llama/Llama-2-70b-hf