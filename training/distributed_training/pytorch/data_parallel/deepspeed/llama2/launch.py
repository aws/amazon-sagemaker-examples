import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import sagemaker,boto3

sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

try:
    role = sagemaker.get_execution_role()
except ValueError:
    iam = boto3.client('iam')
    role = iam.get_role(RoleName='sagemaker_execution_role')['Role']['Arn']
  
print(f"sagemaker role arn: {role}")
print(f"sagemaker bucket: {sagemaker_session.default_bucket()}")
print(f"sagemaker session region: {sagemaker_session.boto_region_name}")

ecr_image = "570106654206.dkr.ecr.us-west-2.amazonaws.com/unified-herring:2.0.1-gpu-py310-cu118-ubuntu20.04-sagemaker"
subnet_config = ["subnet-02bb299d2e6af1d47"]
security_group_config = ["sg-051fa281c203e03a9"]
hyperparameters={
  'model_id': 'meta-llama/Llama-2-7b-chat-hf',
  'gradient_checkpointing': True,
  'bf16': True,
  'optimizer': "adamw_torch",
  'per_device_train_batch_size': 1,
  'epochs': 1,
  'max_steps':50,
  'deepspeed_config':'dsconfig.json'
}

from sagemaker.pytorch import PyTorch
estimator = PyTorch(
  entry_point="train.py",
  base_job_name="llama2-training-smddp",
  role=role,
  image_uri=ecr_image,
  source_dir="code",
  instance_count=2,
  instance_type="ml.p4d.24xlarge",
  sagemaker_session=sagemaker_session,
  subnets=subnet_config,
  hyperparameters=hyperparameters,
  security_group_ids=security_group_config,
  keep_alive_period_in_seconds=600,
  distribution={"torch_distributed": {"enabled": True}},
  debugger_hook_config=False
)

estimator.fit(wait=True)
