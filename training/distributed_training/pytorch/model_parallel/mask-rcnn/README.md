# Use Amazon Sagemaker Distributed Model Parallel to Launch a Mask-RCNN Training Job with Model Parallelization

## Training
In this tutorial we use the SageMaker notebook instance to launch model parallel distributed training jobs with Sagemaker Distributed Model Parallel (SMP) library using [Amazon S3](https://aws.amazon.com/s3/), [Amazon EFS](https://aws.amazon.com/efs/), or [Amazon FSx Lustre](https://aws.amazon.com/fsx/) as data source for training data pipeline. 

This tutorial also offers the option to launch the training job with a custom VPC. Run the script `stack_sm.sh` to create the custom VPC, which also creates the notebook instance and EFS volume under that VPC. The notebook `smp_maskrcnn_tutorial.ipynb` can also create the FSx file system under the same VPC. For the details, please check [Create SageMaker notebook instance in a VPC](https://github.com/aws/amazon-sagemaker-examples/tree/master/advanced_functionality/distributed_tensorflow_mask_rcnn#create-sagemaker-notebook-instance-in-a-vpc) for details.

## Attribution
Much of the code in the `utils/train_script/maskrcnn` is borrowed from: [Nvidia ML-perf Examples](https://github.com/mlcommons/training_results_v0.7/tree/master/NVIDIA/benchmarks/maskrcnn/implementations/pytorch)