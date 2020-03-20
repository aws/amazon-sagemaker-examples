# Deep Graph Library Amazon SageMaker GCN examples for molecular property prediction

In this tutorial, you learn how to run graph convolutional networks (GCNs) for molecular property prediction by using Amazon SageMaker.

With atoms being nodes and bonds being edges, molecules have been an important type of data in the application of 
graph neural networks. In this example, you use the dataset **Tox21**. The 
Toxicology in the 21st Century (Tox21) initiative created a public database measuring the toxicity of compounds. The 
dataset contains qualitative toxicity measurements for 8,014 compounds on 12 different targets, including nuclear 
receptors and stress response pathways. Each target yields a binary classification problem. Therefore, you can model the 
problems as graph classification problems. The molecular benchmark [MoleculeNet](http://moleculenet.ai/) randomly splits the dataset into a training, validation, 
and test set with a 80:10:10 ratio. This tutorial follows that approach.

Use atom descriptors as initial node features. After updating node features as in usual GCN, combine the sum and
maximum of the updated node (atom) representations for graph (molecule) representations. Finally, use a 
feedforward neural network (FNN) to make the predictions from the representations.

For more information about DGL and GCN, see https://docs.dgl.ai

## Setup a Conda environment for DGL (PyTorch backend)

To install a Conda environment for DGL with a GPU-enabled PyTorch backend, use the following steps.
```
# Clone Python3 environment

conda create --name DGL_py36_pytorch1.2_chem --clone python3

# Install PyTorch and DGL
conda install --name DGL_py36_pytorch1.2_chem pytorch=1.2 torchvision -c pytorch
conda install --name DGL_py36_pytorch1.2_chem -c dglteam dgl-cuda10.0=0.4.0
conda install --name DGL_py36_pytorch1.2_chem --update-deps --force libpng
conda install --name DGL_py36_pytorch1.2_chem --update-deps --force -c conda-forge rdkit=2018.09.3
```
You can select DGL_py36_pytorch1.2_chem Conda environment now.
