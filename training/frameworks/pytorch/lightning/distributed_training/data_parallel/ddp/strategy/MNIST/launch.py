import sagemaker

sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

from sagemaker.pytorch import PyTorch
# https://github.com/aws/deep-learning-containers/releases/tag/v1.0-pt-e3-1.12.0-py38
ecr_image="763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:1.12.0-gpu-py38-cu116-ubuntu20.04-e3-v1.0"

# Configure these based on your configuration.
# Refer to following documentation for setting up subnet && security configuration.
# https://github.com/aws/amazon-sagemaker-examples/blob/main/training/distributed_training/pytorch/data_parallel/mnist/pytorch_smdataparallel_mnist_demo.ipynb
# https://github.com/aws/amazon-sagemaker-examples/blob/main/training/distributed_training/pytorch/data_parallel/bert/pytorch_smdataparallel_bert_demo.ipynb
subnet_config = ["subnet-02bb299d2e6af1d47"]
security_group_config = ["sg-051fa281c203e03a9"]

estimator = PyTorch(
  entry_point="mnist.py",
  max_run=1800,
  base_job_name="lightning-ddp-strategy-mnist",
  role=role,
  image_uri=ecr_image,
  source_dir="training_script",
  instance_count=2,
  instance_type="ml.p3dn.24xlarge",
  sagemaker_session=sagemaker_session,
  subnets=subnet_config,
  security_group_ids=security_group_config,
  distribution={"smdistributed": {"dataparallel": {"enabled": True}}},    
  debugger_hook_config=False)

estimator.fit()