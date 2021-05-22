This repository contains examples of deploying Hugging Face models with PyTorch TorchServe on Amazon SageMaker. It includes both single model and multi-model deployments 

Single-model folder contains 2 examples. The first example uses the default SageMaker PyTorch serving container with a requirements.txt file to install the HuggingFace transformers library during inference. The second example extends the default SageMaker PyTorch serving container to include the HuggingFace transformers library. This container can be used at inference time without any further dependencies. Both single model examples use a GPU instance for low latency inference. 

Multi-model folder contains 1 example. It uses the default SageMaker PyTorch serving container and installs the HuggingFace transformers library with a 'pip install' within the inference script. This example shows 2 models deployed on a single CPU instance endpoint leveraging the multi-model capabilities of Amazon SageMaker.

