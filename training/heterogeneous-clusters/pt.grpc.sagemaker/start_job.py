import os
import json
import datetime
import os

import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.instance_group import InstanceGroup
from sagemaker.inputs import TrainingInput

S3_BUCKET_DATASET='sagemaker-us-east-1-776941257690'

IS_CLOUD_JOB = True
IS_HETRO = True
TF_DATA_MODE = 'service' if IS_HETRO else 'local' # local | service
IS_DNN_DISTRIBUTION = False

REGION = 'us-east-1'
os.environ["AWS_DEFAULT_REGION"] = REGION

IS_CLOUD_JOB = True
IS_HETERO = True # if set to false, uses homogenous cluster
PT_DATA_MODE = 'service' if IS_HETERO else 'local' # local | service
IS_DNN_DISTRIBUTION = False # Distributed Training with DNN nodes not tested, set it to False

data_group = InstanceGroup("data_group", "ml.c5.9xlarge", 1) #36 vCPU #change the instance type if IS_HETERO=True
dnn_group = InstanceGroup("dnn_group", "ml.p3.2xlarge", 1)  #8 vCPU #change the instance type if IS_HETERO=True

kwargs = dict()
kwargs['hyperparameters'] = {
    "batch-size": 8192, 
    "num-data-workers": 32, # This number drives the avg. step time. More workers help parallel pre-processing of data. Recommendation: Total no. of cpu 'n' = 'num-data-wokers'+'grpc-workers'+ 2 (reserved)
    "grpc-workers": 2, # No. of workers serving pre-processed data to DNN group (gRPC client). see above formula. 
    "num-dnn-workers": 2, # Modify this no. to be less than the cpu core of your training instances in dnn group
    "pin-memory": True, # Pin to GPU memory
    'iterations' : 100 # No. of iterations in an epoch (must be multiple of 10). 
}

if IS_HETERO:
    kwargs['instance_groups'] = [data_group, dnn_group]
    entry_point='launcher.py'
else:
    kwargs['instance_type'] = 'ml.p3.2xlarge' if IS_CLOUD_JOB else 'local' #change the instance type if IS_HETERO=False
    kwargs['instance_count'] = 1
    entry_point='train.py'
    
if IS_DNN_DISTRIBUTION:
    processes_per_host_dict = {
    'ml.g5.xlarge'  : 1,
    'ml.g5.12xlarge' : 4,
    'ml.p3.8xlarge' : 4,
    'ml.p4d.24xlarge' : 8,
    }
    kwargs['distribution'] = {
        'mpi': {
            'enabled': True,
            'processes_per_host': processes_per_host_dict[dnn_instance_type],
            'custom_mpi_options': '--NCCL_DEBUG INFO'
        },
    }
    if IS_HETERO:
        kwargs['distribution']['instance_groups'] = [dnn_group]

    print(f"distribution={kwargs['distribution']}")

estimator = PyTorch(
    framework_version='1.11.0',    # 1.10.0 or later
    py_version='py38', # Python v3.8 
    role='arn:aws:iam::776941257690:role/service-role/AmazonSageMakerServiceCatalogProductsUseRole',
    entry_point=entry_point,
    source_dir='code',
    volume_size=10,
    max_run=4800,
    disable_profiler=True,
    debugger_hook_config=False,
    **kwargs,
)

s3_input = TrainingInput(
    's3://'+S3_BUCKET_DATASET+'/cifar10-tfrecord/', 
    #instance_groups=['data_group'], # this training channel is created only in data_group instances (i.e., not in dnn_group instance)
    input_mode='FastFile',
    )

data_uri = s3_input if IS_CLOUD_JOB else 'file://./data/'
estimator.fit(
    inputs=data_uri,
    job_name='pt-heterogenous' + 
    '-' + 'H-' + str(IS_HETRO)[0] + 
    '-' + datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
)
