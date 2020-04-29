# Setup Deep Graph Library with Amazon SageMaker to create graph convolutional network examples
The steps here show how to run graph convolutional network (GCN) with Amazon SageMaker. For more information about Deep Graph Library (DGL) and GCN, see the [DGL documentation](https://docs.dgl.ai).

## Setup conda environment for DGL (MXNet backend)
You can install a conda environment for DGL with MXNet backend with a CPU-build.

To create this, use the following steps:
```
# Clone python3 environment
conda create --name DGL_py36_mxnet1.5 --clone python3

# Install MXNet and DGL (This is only CPU version)
source activate DGL_py36_mxnet1.5
conda install -c anaconda scipy
conda install -c anaconda numpy
conda install -c anaconda numexpr
conda install -c anaconda blas=1.0=mkl mkl-service 
conda install -c anaconda mkl_fft==1.0.1 mkl_random==1.0.1
conda install -c anaconda numpy-base==1.16.0 scikit-learn mxnet=1.5.0
conda install -c dglteam dgl
```
You can select DGL_py36_mxnet1.5 conda environment now.

## Setup a conda environment for DGL (PyTorch backend)
You can install a conda environment for DGL with PyTorch backend with GPU-build.

To create this, use the following steps:
```
# Clone python3 environment
conda create --name DGL_py36_pytorch1.2 --clone python3

# Install PyTorch and DGL
conda install --name DGL_py36_pytorch1.2 pytorch=1.2 torchvision -c pytorch
conda install --name DGL_py36_pytorch1.2 -c dglteam dgl-cuda10.0
```
You can select DGL_py36_pytorch1.2 conda environment now.
