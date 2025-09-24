# Serve an NLP model on GPU with Amazon SageMaker Multi-model endpoints (MME)

In this example, we will walk you through how to use NVIDIA Triton Inference Server on Amazon SageMaker MME with GPU feature to deploy two different HuggingFace  **T5** NLP model for **Text Translation**. In particular, this model will be using:

T5-small HuggingFace PyTorch Translation Model (Served using Triton's Python Backend) 

## Steps to run the notebook

1. Launch SageMaker notebook instance with `g5.xlarge` instance. This example can also be run on a SageMaker studio notebook instance but the steps that follow will focus on the notebook instance.
    * IMPORTANT: In Notebook instance settings, within Additional Configuration, for **Volume Size in GB** specify at least **100 GB**.
    * For git repositories select the option `Clone a public git repository to this notebook instance only` and specify the Git repository URL
    
2. Once JupyterLab is ready, launch the **t5_nlp_pytorch_python-gpu.ipynb** notebook with **conda_python3** conda kernel and run through this notebook to learn how to host multiple NLP models on `g5.2xlarge` GPU behind MME endpoint.
