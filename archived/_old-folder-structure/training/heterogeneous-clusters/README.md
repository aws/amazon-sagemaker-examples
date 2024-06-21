# SageMaker Heterogeneous Clusters for Model Training
In July 2022, we [launched](https://aws.amazon.com/about-aws/whats-new/2022/07/announcing-heterogeneous-clusters-amazon-sagemaker-model-training/) heterogeneous clusters for Amazon SageMaker
model training, which enables you to launch training jobs that use different instance types and
families in a single job. A primary use case is offloading data preprocessing to 
compute-optimized instance types, whereas the deep neural network (DNN) process continues to
run on GPU or ML accelerated instance types. 

In this repository, you'll find TensorFlow (tf.data.service) and PyTorch (a custom gRPC based distributed data loading) examples which demonstrates how to use heterogeneous clusters in your SageMaker training jobs. You can use these examples with minimal code changes in your existing training scripts.

![Hetero job diagram](tf.data.service.sagemaker/images/basic-heterogeneous-job.png)

## Examples:

### Hello world example
- [**Heterogeneous Clusters - a hello world example**](hello.world.sagemaker/helloworld-example.ipynb):
This basic example runs a heterogeneous training job consisting of two instance groups. Each group includes a different instance_type. 
Each instance prints its instance group information and exits. 
Note: This example only shows how to orchestrate the training job with instance type. For actual code to help with a distributed data loader, see the TensorFlow or PyTorch examples below.

### TensorFlow examples
- [**TensorFlow's tf.data.service based Amazon SageMaker Heterogeneous Clusters**](tf.data.service.sagemaker/hetero-tensorflow-restnet50.ipynb):
This TensorFlow example runs both Homogeneous and Heterogeneous clusters SageMaker training job, and compares their results. The heterogeneous cluster training job runs with two instance groups:
  - `data_group` - this group has two ml.c5.18xlarge instances to which data preprocessing/augmentation is offloaded.
  - `dnn_group` - this group has one ml.p4d.24xlarge instance (8GPUs) in a horovod/MPI distribution.

### PyTorch examples
- [**PyTorch and gRPC distributed dataloader based Amazon SageMaker Heterogeneous Clusters**](pt.grpc.sagemaker/hetero-pytorch-mnist.ipynb):
This PyTorch example enables you to run both Homogeneous and Heterogeneous clusters SageMaker training job. We then compare their results, and understand price performance benefits. 
  - `data_group` - this group has one ml.c5.9xlarge instance for offloading data preprocessing job.
  - `dnn_group` - this group has one ml.p3.2xlarge instance
