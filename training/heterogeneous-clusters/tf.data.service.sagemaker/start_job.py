import datetime
from sagemaker.tensorflow import TensorFlow
from sagemaker.instance_group import InstanceGroup
import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--tf_data_mode', type=str, default='local', 
    help="'service' distributed dataset using tf.data.service. 'local' use standard tf.data")
parser.add_argument('--is_cloud_job', default=True, action=argparse.BooleanOptionalAction,
    help="True to run in the cloud, False to run on local machine")
parser.add_argument('--is_hetero', default=True, action=argparse.BooleanOptionalAction,
    help="True to run in the heterogeneous mode (GPU + CPU instances), False when running in the homogeneous mode (GPU instances only)")
parser.add_argument("--num_of_data_workers", type=int, default=1)
parser.add_argument("--num_of_data_instances", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=1024)

parser.add_argument("--training_dir", type=str, default='data')
args = parser.parse_args()

assert args.is_cloud_job or not args.is_hetero, 'Heterogeneous cluster is not supported in sagemaker local mode'
assert args.is_hetero or args.tf_data_mode == 'local', 'TODO: tf.data.service not implemented in homogeneous cluster yet'

REGION = 'us-east-1'
os.environ["AWS_DEFAULT_REGION"] = REGION
dnn_instance_type = 'ml.p4d.24xlarge' if args.is_cloud_job else 'local_gpu' # @see: https://aws.amazon.com/sagemaker/pricing/
data_instance_type = "ml.c5.18xlarge"

# Group for CPU instances that will run tf.data.service dispatcher/workers processes.
data_group = InstanceGroup("data_group", data_instance_type, args.num_of_data_instances) if args.is_hetero else None
# Group for deep neural network (dnn) with accleartors (e.g., GPU, FPGA, etc.)
dnn_group =  InstanceGroup("dnn_group", dnn_instance_type, 1) if args.is_hetero else None

kwargs = dict()
kwargs['hyperparameters'] = {
    'epochs' : 3,
    'steps_per_epoch' : 500,
    'num_of_data_workers' : args.num_of_data_workers, # How many tf.data.server Workers to start
    'batch_size' : args.batch_size,
    'tf_data_mode' : args.tf_data_mode,
}

if args.is_hetero:
    print(f'args.is_hetero = {args.is_hetero}')
    kwargs['instance_groups'] = [data_group, dnn_group]    
else:
    kwargs['instance_type'] = dnn_instance_type if args.is_cloud_job else 'local'
    kwargs['instance_count'] = 1

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
        'custom_mpi_options': '--NCCL_DEBUG WARN'
    },
}
if args.is_hetero:
    # Start an MPI cluster only DNN instance group only
    kwargs['distribution']['instance_groups'] = [dnn_group] # type: ignore 

print(f"distribution={kwargs['distribution']}")

estimator = TensorFlow(
    entry_point='launcher.py',
    source_dir='./code',
    framework_version='2.9.1',
    py_version='py39',
    role=os.environ.get('SAGEMAKER_ROLE'),
    volume_size=30,
    max_run=1800,
    disable_profiler=True,
    **kwargs,
)
print(f'kwargs={kwargs}')
estimator.fit(
    job_name=f'hetero-tf-data-{args.tf_data_mode}-Dnode{args.num_of_data_instances}-wrkrs-{args.num_of_data_workers}-{datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")}',
)
