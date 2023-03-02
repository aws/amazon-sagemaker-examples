# Serve an ResNet-50 ONNX model along with PyTorch and TensorRT models on GPU with Amazon SageMaker Multi-model endpoints (MME)

In this example, we will walk you through how to use NVIDIA Triton Inference Server on Amazon SageMaker MME with GPU
to deploy Resnet-50 ONNX, TensorRT and Pytorch model for **Image Classification**.

## Steps to run the notebook

1. Launch SageMaker notebook instance with `g4dn.xlarge` instance. This example can also be run on a SageMaker studio notebook instance but the steps that follow will focus on the notebook instance.
   
    * For git repositories select the option `Clone a public git repository to this notebook instance only` and specify the Git repository URL
    
2. Once JupyterLab is ready, launch the **resnet_onnx_pytorch_tensorRT_backend_MME_triton.ipynb** notebook with
**conda_python3** conda kernel and run through this notebook to learn how to host multiple CV models on `g4dn.xlarge`
GPU behind MME endpoint. Notice that due to the sizes of the models, the first time the model invokes take seconds, but
the second time it takes milliseconds. You can also run in a larger GPU instance like g5.xlarge to see the difference.
