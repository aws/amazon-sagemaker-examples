import datetime
from sagemaker.tensorflow import TensorFlow
from sagemaker.instance_group import InstanceGroup
import os

REGION = 'us-east-1'
os.environ["AWS_DEFAULT_REGION"] = REGION

# https://aws.amazon.com/sagemaker/pricing/
data_group = InstanceGroup("data_group", "ml.c5.xlarge", 1)
dnn_group = InstanceGroup("dnn_group", "ml.m4.xlarge", 1)  

estimator = TensorFlow(
    entry_point='train.py',
    source_dir='./source_dir',
    #instance_type='ml.m4.xlarge',
    #instance_count=1,
    instance_groups = [data_group, dnn_group,],
    framework_version='2.9.1',
    py_version='py39',
    role=os.environ.get('SAGEMAKER_ROLE'),
    volume_size=10,
    max_run=3600,
    max_wait=3600,
    disable_profiler=True,
    #use_spot_instances=True,
)

estimator.fit(
    job_name='hello-world-heterogenous' + 
    '-' + datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
)
