# Serve an ResNet-50 ONNX model on GPU with Amazon SageMaker endpoint (SME) with Triton

In this example, we will walk you through how to use NVIDIA Triton Inference Server on Amazon SageMaker SME with GPU to deploy Resnet-50 ONNX for **Image Classification**.

## Steps to run the notebook

1. Launch SageMaker notebook instance with `g4dn.xlarge` instance. This example can also be run on a SageMaker studio notebook instance but the steps that follow will focus on the notebook instance.
   
    * For git repositories select the option `Clone a public git repository to this notebook instance only` and specify the Git repository URL
    
2. Once JupyterLab is ready, launch the **resnet_onnx_backend_SME_triton_v2.ipynb** notebook with
**conda_python3** conda kernel and run through this notebook to learn how to host multiple CV models on `g4dn.xlarge`
GPU behind SME endpoint. 

Note This notebook was tested with the `conda_pytorch_p39` kernel on an Amazon SageMaker notebook instance of type `g4dn.xlarge`. It is a modified version of the original version of [this](https://github.com/aws/amazon-sagemaker-examples/blob/main/multi-model-endpoints/mme-on-gpu/cv/resnet50_mme_with_gpu.ipynb) sample notebook Here by [Vikram Elango](https://github.com/vikramelango).
