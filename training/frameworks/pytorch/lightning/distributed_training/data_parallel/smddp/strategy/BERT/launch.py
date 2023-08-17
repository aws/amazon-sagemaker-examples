import sagemaker
from sagemaker.pytorch import PyTorch

sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

# https://github.com/aws/deep-learning-containers/blob/master/available_images.md
ecr_image="763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:1.12.0-gpu-py38-cu113-ubuntu20.04-sagemaker"

estimator = PyTorch(
  entry_point="bert.py",
  max_run=1800,
  base_job_name="lightning-smddp-strategy-bert",
  role=role,
  image_uri=ecr_image,
  source_dir="training_script",
  instance_count=2,
  instance_type="ml.p3dn.24xlarge",
  sagemaker_session=sagemaker_session,
  distribution={"smdistributed": {"dataparallel": {"enabled": True}}},    
  debugger_hook_config=False)

estimator.fit()