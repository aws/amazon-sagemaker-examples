# Serve an ResNet Pytorch model on GPU with Amazon SageMaker Multi-model endpoints (MME)

In this example, we will walk you through how to use NVIDIA Triton Inference Server on Amazon SageMaker MME with GPU feature to deploy Resnet Pytorch model for **Image Classification**. 

## Steps to run the notebook

1. Launch SageMaker notebook instance with `g5.xlarge` instance. This example can also be run on a SageMaker studio notebook instance but the steps that follow will focus on the notebook instance.
   
    * For git repositories select the option `Clone a public git repository to this notebook instance only` and specify the Git repository URL
    
2. Once JupyterLab is ready, launch the **resnet_pytorch_python_backend_MME.ipynb** notebook with **conda_python3** conda kernel and run through this notebook to learn how to host multiple CV models on `g5.2xlarge` GPU behind MME endpoint.

