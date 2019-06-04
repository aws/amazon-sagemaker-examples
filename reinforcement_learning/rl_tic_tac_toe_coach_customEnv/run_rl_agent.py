import sagemaker
import os
from sagemaker.rl import RLEstimator, RLToolkit, RLFramework


# Setup
sess = sagemaker.session.Session()
s3_bucket = sess.default_bucket()
s3_output_path = 's3://{}/'.format(s3_bucket)
job_name_prefix = 'rl-tic-tac-toe'
local_mode = False
role = sagemaker.get_execution_role()


# Run RL
estimator = RLEstimator(source_dir='src',
                        entry_point="train-coach.py",
                        dependencies=["common/sagemaker_rl"],
                        toolkit=RLToolkit.COACH,
                        toolkit_version='0.11.0',
                        framework=RLFramework.TENSORFLOW,
                        role=role,
                        train_instance_count=1,
                        train_instance_type='ml.m5.xlarge',
                        output_path=s3_output_path,
                        base_job_name=job_name_prefix,
                        hyperparameters = {"RLCOACH_PRESET" : "preset"})


estimator.fit()



# Read in output
job_name = estimator.latest_training_job.job_name

os.system('rm -rf output/')
os.system('mkdir output')
os.system('aws s3 cp {}{}/output/output.tar.gz ./output/'.format(s3_output_path, job_name))
os.system('cd output; tar xzvf output.tar.gz; cd ../')
