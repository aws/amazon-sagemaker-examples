import boto3
import sagemaker
import sys
import os
import re
import numpy as np
import subprocess

sys.path.append("common")
from botocore.errorfactory import ClientError
import constant as const
from misc import get_execution_role, wait_for_s3_object
from docker_utils import build_and_push_docker_image
from sagemaker.rl import RLEstimator, RLToolkit, RLFramework
from copy_to_sagemaker_container import get_sagemaker_docker, copy_to_sagemaker_container, get_custom_image_name
from time import gmtime, strftime
import time
from IPython.display import Markdown
from markdown_helper import *

import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import display, clear_output

import warnings

warnings.filterwarnings("ignore")


class DeepRacerEngine:
    # Select the instance type
    instance_type = None

    # Starting SageMaker session
    sage_session = None

    # Create unique job name.
    job_name_prefix = None

    # Duration of job in seconds (1 hours)
    job_duration_in_seconds = None

    # Track name
    track_name = None

    # IAM Params
    sagemaker_role = None

    # Metric Defs
    metric_definitions = None

    # Docker Image Varaibles
    custom_image_name = None
    repository_short_name = None
    sagemaker_docker_id = None
    docker_build_args = None

    # VPC params
    ec2 = None
    deepracer_security_groups = None
    deepracer_vpc = None
    deepracer_subnets = None
    route_tables = None

    # S3 variavles
    s3_location = None
    s3_output_path = None

    # robmaker parms
    estimator = None
    job_name = None

    # param Kinesis Video Stream
    kvs_stream_name = None

    # robomaker Parsm
    robomaker = None
    robomaker_s3_key = None
    robomaker_source = None
    simulation_software_suite = None
    robot_software_suite = None
    rendering_engine = None
    app_name = None
    response = None
    simulation_app_arn = None

    # Simulationb Parsms
    num_simulation_workers = None
    envriron_vars = None
    simulation_application = None
    vpcConfig = None
    responses = None
    job_arns = None
    simulation_application_bundle_location = None

    # output Parsm
    tmp_dir = None
    training_metrics_file = None
    training_metrics_path = None

    # evaluate parsms:
    eval_envriron_vars = None
    eval_simulation_application = None
    eval_vpcConfig = None
    eval_responses = None

    # Eval output
    evaluation_metrics_file = None
    evaluation_metrics_path = None
    evaluation_trials = None

    # AWS Region
    aws_region = None

    ''' 
    INIT
    '''

    def __init__(self, kwargs):

        print('***Deep Racer Engine Backend***')

        if 'job_name' in kwargs:
            self.job_name = kwargs['job_name']
        else:
            raise Exception("A Job Name MUST be provided. Stopping execution.")

        if 'instance_type' in kwargs:
            self.instance_type = kwargs['instance_type']
        else:
            self.instance_type = const.default_instance_type

        if 'instance_pool_count' in kwargs:
            self.instance_pool_count = kwargs['instance_pool_count']
        else:
            self.instance_pool_count = const.default_instance_pool

        if 'job_duration' in kwargs:
            self.job_duration_in_seconds = kwargs['job_duration']
        else:
            self.job_duration_in_seconds = const.default_job_duration

        if 'track_name' in kwargs:
            self.track_name = kwargs['track_name']
        else:
            self.track_name = const.default_track_name

        if 'racetrack_env' in kwargs:
            self.envir_file_local = kwargs['racetrack_env']
        else:
            self.envir_file_local = const.envir_file_local

        if 'reward_policy' in kwargs:
            self.reward_file_local = kwargs['reward_policy']
        else:
            self.reward_file_local = const.reward_file_local

        if 'meta_file' in kwargs:
            self.model_meta_file_local = kwargs['meta_file']
        else:
            self.model_meta_file_local = const.model_meta_file_local

        if 'presets' in kwargs:
            self.presets_file_local = kwargs['presets']
        else:
            self.presets_file_local = const.presets_file_local

        if 'custom_action_space' in kwargs:
            if ('min_speed' in kwargs) and ('max_speed' in kwargs) and (
                    'min_steering_angle' in kwargs) and ('max_steering_angle' in kwargs):
                
                # Create the actionspace...
                self.create_actionspace(kwargs)
                self.model_meta_file_local = const.model_meta_file_custom_local
            else:
                self.model_meta_file_local = const.model_meta_file_local

        else:
            self.model_meta_file_local = const.model_meta_file_local

        if 'evaluation_trials' in kwargs:
            self.evaluation_trials = kwargs['evaluation_trials']
        else:
            self.evaluation_trials = const.evaluation_trials

        # local file where hyperparams will be saved..
        self.presets_hyperp_local = const.tmp_hyperparam_preset

        self.create_hyperparams(kwargs)

        self.sage_session = sagemaker.session.Session()
        self.s3 = boto3.resource('s3')
        self.job_name_prefix = self.job_name

        if self.track_name not in const.track_name:
            raise Exception("The track name provded does not exist. Please provide a trackname which matches an"
                            "available track")

        self.aws_region = self.sage_session.boto_region_name
        if self.aws_region not in ["us-west-2", "us-east-1", "eu-west-1"]:
            raise Exception("This notebook uses RoboMaker which is available only in US East (N. Virginia),"
                            "US West (Oregon) and EU (Ireland). Please switch to one of these regions.")

    def create_hyperparams(self, kwargs):
        print('>>Creating Custom Hyperparameters ')
        # first we're going to get all the global variables
        self.hyperparam_data = {}
        with open(const.default_hyperparam_preset) as fp:
            self.hyperparam_data = json.load(fp)
            if 'learning_rate' in kwargs:
                self.hyperparam_data['learning_rate'] = kwargs['learning_rate']

            if 'batch_size' in kwargs:
                self.hyperparam_data['batch_size'] = kwargs['batch_size']

            if 'optimizer_epsilon' in kwargs:
                self.hyperparam_data['optimizer_epsilon'] = kwargs['optimizer_epsilon']

            if 'optimization_epochs' in kwargs:
                self.hyperparam_data['optimization_epochs'] = kwargs['optimization_epochs']

            # now write these key,values to file
            with open(const.tmp_hyperparam_preset, 'w') as filew:
                for k, v in self.hyperparam_data.items():
                    c = '{}={}\n'.format(k, v)
                    filew.write(c)

    def create_actionspace(self, kwargs):
        print('>>Creating Custom Action space')
        self.action_space_min_speed = kwargs['min_speed']
        self.action_space_max_speed = kwargs['max_speed']
        self.action_space_min_steering_angle = kwargs['min_steering_angle']
        self.action_space_max_steering_angle = kwargs['max_steering_angle']
        #Optional...
        if ('speed_interval' in kwargs) and ('steering_angle_interval' in kwargs):
            self.action_space_speed_interval = kwargs['speed_interval']
            self.action_space_steering_angle_interval = kwargs['steering_angle_interval']
        else:
            self.action_space_speed_interval = 1
            self.action_space_steering_angle_interval = 5
        
        min_speed = self.action_space_min_speed
        max_speed = self.action_space_max_speed
        speed_interval = self.action_space_speed_interval

        min_steering_angle = self.action_space_min_steering_angle
        max_steering_angle = self.action_space_max_steering_angle
        steering_angle_interval = self.action_space_steering_angle_interval

        output = {"action_space": []}
        index = 0
        speed = min_speed
        while speed <= max_speed:
            steering_angle = min_steering_angle
            while steering_angle <= max_steering_angle:
                output["action_space"].append({"index": index,
                                               "steering_angle": steering_angle,
                                               "speed": speed}
                                              )
                steering_angle += steering_angle_interval
                index += 1
            speed += speed_interval

        # now write these key,values to file
        with open(const.model_meta_file_custom_local, 'w') as filew:
            json.dump(output, filew)
        

    def configure_environment(self):

        print('********************************')
        print('PERFORMING ALL DOCKER, VPC, AND ROUTING TABLE WORK....')
        ## Configure The S3 Bucket which this job will work for
        self.configure_s3_bucket()
        ## Configure the IAM role and ensure that the correct access priv are available
        self.create_iam_role()

        self.build_docker_container()

        self.configure_vpc()

        self.create_routing_tables()

        self.upload_environments_and_rewards_to_s3()

        self.configure_metrics()

        self.configure_estimator()

        self.configure_kinesis_stream()

        self.start_robo_maker()

    ''' 
    TO-ADD
    '''

    def start_training_testing_process(self):

        print('********************************')
        print('PERFORMING ALL DOCKER, VPC, AND ROUTING TABLE WORK....')
        ## Configure The S3 Bucket which this job will work for
        self.configure_s3_bucket()
        ## Configure the IAM role and ensure that the correct access priv are available
        self.create_iam_role()

        self.build_docker_container()

        self.configure_vpc()

        self.create_routing_tables()

        self.upload_environments_and_rewards_to_s3()

        self.configure_metrics()

        self.configure_estimator()

        self.configure_kinesis_stream()

        self.start_robo_maker()

        self.create_simulation_application()

        self.start_simulation_job()

    #         self.plot_training_output()

    def start_model_evaluation(self):

        self.configure_evaluation_process()

        self.plot_evaluation_output()

    '''
    Set up the linkage and authentication to the S3 bucket that we want to use for checkpoint and metadata.    
    '''

    def configure_s3_bucket(self):

        # S3 bucket
        self.s3_bucket = self.sage_session.default_bucket()

        # SDK appends the job name and output folder
        self.s3_output_path = 's3://{}/'.format(self.s3_bucket)

        # Ensure that the S3 prefix contains the keyword 'sagemaker'
        self.s3_prefix = self.job_name_prefix + "-sagemaker-" + strftime("%y%m%d-%H%M%S", gmtime())

        # Get the AWS account id of this account
        self.sts = boto3.client("sts")
        self.account_id = self.sts.get_caller_identity()['Account']

        print("Using s3 bucket {}".format(self.s3_bucket))
        print("Model checkpoints and other metadata will be stored at: \ns3://{}/{}".format(self.s3_bucket,
                                                                                            self.s3_prefix))

    def create_iam_role(self):
        try:
            self.sagemaker_role = sagemaker.get_execution_role()
        except:
            self.sagemaker_role = get_execution_role('sagemaker')

        print("Using Sagemaker IAM role arn: \n{}".format(self.sagemaker_role))

    def build_docker_container(self):
        cpu_or_gpu = 'gpu' if self.instance_type.startswith('ml.p') else 'cpu'
        self.repository_short_name = "sagemaker-docker-%s" % cpu_or_gpu
        self.custom_image_name = get_custom_image_name(self.repository_short_name)
        try:
            print("Copying files from your notebook to existing sagemaker container")
            self.sagemaker_docker_id = get_sagemaker_docker(self.repository_short_name)
            copy_to_sagemaker_container(self.sagemaker_docker_id, self.repository_short_name)
        except Exception as e:
            print("Creating sagemaker container")
            self.docker_build_args = {
                'CPU_OR_GPU': cpu_or_gpu,
                'AWS_REGION': boto3.Session().region_name,
            }
            self.custom_image_name = build_and_push_docker_image(self.repository_short_name,
                                                                 build_args=self.docker_build_args)
            print("Using ECR image %s" % self.custom_image_name)

    def configure_vpc(self):

        self.ec2 = boto3.client('ec2')

        #
        # Check if the user has Deepracer-VPC and use that if its present. This will have all permission.
        # This VPC will be created when you have used the Deepracer console and created one model atleast
        # If this is not present. Use the default VPC connnection
        #
        self.deepracer_security_groups = [group["GroupId"] for group in
                                          self.ec2.describe_security_groups()['SecurityGroups'] \
                                          if group['GroupName'].startswith("aws-deepracer-")]

        # deepracer_security_groups = False
        if (self.deepracer_security_groups):
            print("Using the DeepRacer VPC stacks. This will be created if you run one training job from console.")
            self.deepracer_vpc = [vpc['VpcId'] for vpc in self.ec2.describe_vpcs()['Vpcs'] \
                                  if "Tags" in vpc for val in vpc['Tags'] \
                                  if val['Value'] == 'deepracer-vpc'][0]
            self.deepracer_subnets = [subnet["SubnetId"] for subnet in self.ec2.describe_subnets()["Subnets"] \
                                      if subnet["VpcId"] == self.deepracer_vpc]
        else:
            print("Using the default VPC stacks")
            self.deepracer_vpc = [vpc['VpcId'] for vpc in self.ec2.describe_vpcs()['Vpcs'] if vpc["IsDefault"] == True][
                0]

            self.deepracer_security_groups = [group["GroupId"] for group in
                                              self.ec2.describe_security_groups()['SecurityGroups'] \
                                              if 'VpcId' in group and group["GroupName"] == "default" and group[
                                                  "VpcId"] == self.deepracer_vpc]

            self.deepracer_subnets = [subnet["SubnetId"] for subnet in self.ec2.describe_subnets()["Subnets"] \
                                      if subnet["VpcId"] == self.deepracer_vpc and subnet['DefaultForAz'] == True]

        print("Using VPC:", self.deepracer_vpc)
        print("Using security group:", self.deepracer_security_groups)
        print("Using subnets:", self.deepracer_subnets)

    '''
    A SageMaker job running in VPC mode cannot access S3 resourcs. 
    So, we need to create a VPC S3 endpoint to allow S3 access from SageMaker container. 
    To learn more about the VPC mode, 
    please visit [this link.](https://docs.aws.amazon.com/sagemaker/latest/dg/train-vpc.html)
    '''

    def create_routing_tables(self):

        print("Creating Routing Tables")
        try:
            self.route_tables = [route_table["RouteTableId"] for route_table in
                                 self.ec2.describe_route_tables()['RouteTables'] \
                                 if route_table['VpcId'] == self.deepracer_vpc]
        except Exception as e:
            if "UnauthorizedOperation" in str(e):
                # display(Markdown(generate_help_for_s3_endpoint_permissions(self.sagemaker_role)))
                print(e, 'UnauthorizedOperation')
            else:
                print('EE')
                # display(Markdown(create_s3_endpoint_manually(self.aws_region, self.deepracer_vpc)))
            raise e

        print("Trying to attach S3 endpoints to the following route tables:", self.route_tables)

        if not self.route_tables:
            raise Exception(("No route tables were found. Please follow the VPC S3 endpoint creation "
                             "guide by clicking the above link."))
        try:
            self.ec2.create_vpc_endpoint(DryRun=False,
                                         VpcEndpointType="Gateway",
                                         VpcId=self.deepracer_vpc,
                                         ServiceName="com.amazonaws.{}.s3".format(self.aws_region),
                                         RouteTableIds=self.route_tables)
            print("S3 endpoint created successfully!")
        except Exception as e:
            if "RouteAlreadyExists" in str(e):
                print("S3 endpoint already exists.")
            elif "UnauthorizedOperation" in str(e):
                # display(Markdown(generate_help_for_s3_endpoint_permissions(role)))
                raise e
            else:
                # display(Markdown(create_s3_endpoint_manually(aws_region, deepracer_vpc)))
                raise e

    def upload_environments_and_rewards_to_s3(self):

        self.s3_location = self.s3_prefix
        print(self.s3_location)

        # Clean up the previously uploaded files
        bucket = self.s3.Bucket(self.s3_bucket)
        bucket.objects.filter(Prefix=self.s3_prefix).delete()
        # !aws s3 rm --recursive {s3_location}

        # Make any changes to the environment and preset files below and upload these files
        # !aws s3 cp src/markov/environments/deepracer_racetrack_env.py {self.s3_location}/environments/deepracer_racetrack_env.py
        envir_file_s3 = self.s3_location + '/environments/deepracer_racetrack_env.py'
        bucket.upload_file(self.envir_file_local, envir_file_s3)

        # !aws s3 cp src/markov/rewards/complex_reward.py {s3_location}/rewards/reward_function.py
        reward_file_s3 = self.s3_location + '/rewards/reward_function.py'
        bucket.upload_file(self.reward_file_local, reward_file_s3)

        # !aws s3 cp src/markov/actions/model_metadata_10_state.json {s3_location}/model_metadata.json
        model_meta_file_s3 = self.s3_location + '/model_metadata.json'
        bucket.upload_file(self.model_meta_file_local, model_meta_file_s3)

        # !aws s3 cp src/markov/presets/default.py {s3_location}/presets/preset.py
        presets_file_s3 = self.s3_location + '/presets/preset.py'
        bucket.upload_file(self.presets_file_local, presets_file_s3)

        presets_hyperparams_file_s3 = self.s3_location + '/presets/preset_hyperparams.py'
        bucket.upload_file(const.tmp_hyperparam_preset, presets_hyperparams_file_s3)
        print('Cleaning Up Tmp HyperParam file')
        os.remove(const.tmp_hyperparam_preset)

    def configure_metrics(self):

        self.metric_definitions = [
            # Training> Name=main_level/agent, Worker=0, Episode=19, Total reward=-102.88, Steps=19019, Training iteration=1
            {'Name': 'reward-training',
             'Regex': '^Training>.*Total reward=(.*?),'},

            # Policy training> Surrogate loss=-0.32664725184440613, KL divergence=7.255815035023261e-06, Entropy=2.83156156539917, training epoch=0, learning_rate=0.00025
            {'Name': 'ppo-surrogate-loss',
             'Regex': '^Policy training>.*Surrogate loss=(.*?),'},
            {'Name': 'ppo-entropy',
             'Regex': '^Policy training>.*Entropy=(.*?),'},

            # Testing> Name=main_level/agent, Worker=0, Episode=19, Total reward=1359.12, Steps=20015, Training iteration=2
            {'Name': 'reward-testing',
             'Regex': '^Testing>.*Total reward=(.*?),'},
        ]

    def configure_estimator(self):
        self.estimator = RLEstimator(entry_point=const.entry_point,
                                     source_dir=const.source_dir,
                                     image_name=self.custom_image_name,
                                     dependencies=["common/"],
                                     role=self.sagemaker_role,
                                     train_instance_type=self.instance_type,
                                     train_instance_count=self.instance_pool_count,
                                     output_path=self.s3_output_path,
                                     base_job_name=self.job_name_prefix,
                                     metric_definitions=self.metric_definitions,
                                     train_max_run=self.job_duration_in_seconds,
                                     hyperparameters={
                                         "s3_bucket": self.s3_bucket,
                                         "s3_prefix": self.s3_prefix,
                                         "aws_region": self.aws_region,
                                         "preset_s3_key": "%s/presets/preset.py" % self.s3_prefix,
                                         "model_metadata_s3_key": "%s/model_metadata.json" % self.s3_prefix,
                                         "environment_s3_key": "%s/environments/deepracer_racetrack_env.py" % self.s3_prefix,
                                         "batch_size": self.hyperparam_data['batch_size'],
                                         "num_epochs": self.hyperparam_data['optimization_epochs'],
                                         "beta_entropy": self.hyperparam_data['beta_entropy'],
                                         "lr": self.hyperparam_data['learning_rate'],
                                         "num_episodes_between_training": 20,
                                         "discount_factor": self.hyperparam_data['discount']
                                     },
                                     subnets=self.deepracer_subnets,
                                     security_group_ids=self.deepracer_security_groups,
                                     )

        self.estimator.fit(wait=False)
        self.job_name = self.estimator.latest_training_job.job_name
        print("Training job: %s" % self.job_name)

    def configure_kinesis_stream(self):

        self.kvs_stream_name = "dr-kvs-{}".format(self.job_name)

        self.kinesis = boto3.client('kinesis')

        res = self.kinesis.create_stream(
            StreamName=self.kvs_stream_name,
            ShardCount=10
        )

        #         res = self.kinesis.decrease_stream_retention_period(
        #             StreamName=self.kvs_stream_name,
        #             RetentionPeriodHours=8
        #         )

        # !aws --region {aws_region} kinesisvideo create-stream --stream-name {kvs_stream_name} --media-type video/h264 --data-retention-in-hours 24
        print ("Created kinesis video stream {}".format(self.kvs_stream_name))

    #         print(res)

    def start_robo_maker(self):
        self.robomaker = boto3.client("robomaker")

    def create_simulation_application(self):
        self.robomaker_s3_key = 'robomaker/simulation_ws.tar.gz'
        self.robomaker_source = {'s3Bucket': self.s3_bucket,
                                 's3Key': self.robomaker_s3_key,
                                 'architecture': "X86_64"}
        self.simulation_software_suite = {'name': 'Gazebo',
                                          'version': '7'}
        self.robot_software_suite = {'name': 'ROS',
                                     'version': 'Kinetic'}
        self.rendering_engine = {'name': 'OGRE',
                                 'version': '1.x'}

        bucket = self.s3.Bucket(self.s3_bucket)

        download = False
        try:
            self.s3.Object(self.s3_bucket, self.robomaker_s3_key).load()
        except ClientError:
            download = True

        # for now we will always download!
        if download:
            if not os.path.exists('./build/output.tar.gz'):
                print("Using the latest simapp from public s3 bucket")

                # Download Robomaker simApp for the deepracer public s3 bucket
                copy_source = {
                    'Bucket': 'deepracer-managed-resources-us-east-1',
                    'Key': 'deepracer-simapp-notebook.tar.gz'
                }

                simulation_application_bundle_s3 = './'
                self.s3.Bucket('deepracer-managed-resources-us-east-1').download_file(
                    'deepracer-simapp-notebook.tar.gz',
                    './deepracer-simapp-notebook.tar.gz')

                # Remove if the Robomaker sim-app is present in s3 bucket
                sim_app_filename = self.robomaker_s3_key
                bucket.delete_objects(
                    Delete={
                        'Objects': [
                            {
                                'Key': sim_app_filename
                            },
                        ],
                        'Quiet': True
                    })

                # Uploading the Robomaker SimApp to your S3 bucket
                simulation_application_bundle_location = "./deepracer-simapp-notebook.tar.gz"
                simulation_application_bundle_s3 = self.robomaker_s3_key
                bucket.upload_file(simulation_application_bundle_location, simulation_application_bundle_s3)

                # Cleanup the locally downloaded version of SimApp
                sim_app_filename = './deepracer-simapp-notebook.tar.gz'
                os.remove(sim_app_filename)

            else:
                print("Using the simapp from build directory")
                # !aws s3 cp ./build/output.tar.gz s3://{self.s3_bucket}/{self.robomaker_s3_key}
                sim_app_build_location = "./build/output.tar.gz"
                sim_app_build_s3 = self.robomaker_s3_key
                bucket.upload_file(sim_app_build_location, sim_app_build_s3)

        self.app_name = "deepracer-notebook-application" + strftime("%y%m%d-%H%M%S", gmtime())

        print('App Name: {}'.format(self.app_name))

        try:
            self.response = self.robomaker.create_simulation_application(name=self.app_name,
                                                                         sources=[self.robomaker_source],
                                                                         simulationSoftwareSuite=self.simulation_software_suite,
                                                                         robotSoftwareSuite=self.robot_software_suite,
                                                                         renderingEngine=self.rendering_engine)
            self.simulation_app_arn = self.response["arn"]
            print("Created a new simulation app with ARN:", self.simulation_app_arn)
        except Exception as e:
            if "AccessDeniedException" in str(e):
                # display(Markdown(generate_help_for_robomaker_all_permissions(role)))
                raise e
            else:
                raise e

    def start_simulation_job(self):
        self.num_simulation_workers = 1

        self.training_metrics_file = "{}/training_metrics.json".format(self.s3_prefix)

        self.envriron_vars = {
            "WORLD_NAME": self.track_name,
            "KINESIS_VIDEO_STREAM_NAME": self.kvs_stream_name,
            "SAGEMAKER_SHARED_S3_BUCKET": self.s3_bucket,
            "SAGEMAKER_SHARED_S3_PREFIX": self.s3_prefix,
            "TRAINING_JOB_ARN": self.job_name,
            "APP_REGION": self.aws_region,
            "METRIC_NAME": "TrainingRewardScore",
            "METRIC_NAMESPACE": "AWSDeepRacer",
            "REWARD_FILE_S3_KEY": "%s/rewards/reward_function.py" % self.s3_prefix,
            "MODEL_METADATA_FILE_S3_KEY": "%s/model_metadata.json" % self.s3_prefix,
            "METRICS_S3_BUCKET": self.s3_bucket,
            "METRICS_S3_OBJECT_KEY": self.training_metrics_file,
            "TARGET_REWARD_SCORE": "None",
            "NUMBER_OF_EPISODES": "0",
            "ROBOMAKER_SIMULATION_JOB_ACCOUNT_ID": self.account_id
        }

        self.simulation_application = {"application": self.simulation_app_arn,
                                       "launchConfig": {"packageName": "deepracer_simulation_environment",
                                                        "launchFile": "distributed_training.launch",
                                                        "environmentVariables": self.envriron_vars}
                                       }

        self.vpcConfig = {"subnets": self.deepracer_subnets,
                          "securityGroups": self.deepracer_security_groups,
                          "assignPublicIp": True}

        self.responses = []
        for job_no in range(self.num_simulation_workers):
            client_request_token = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
            response = self.robomaker.create_simulation_job(iamRole=self.sagemaker_role,
                                                            clientRequestToken=client_request_token,
                                                            maxJobDurationInSeconds=self.job_duration_in_seconds,
                                                            failureBehavior="Continue",
                                                            simulationApplications=[self.simulation_application],
                                                            vpcConfig=self.vpcConfig
                                                            )
            self.responses.append(response)

        print("Created the following jobs:")
        self.job_arns = [response["arn"] for response in self.responses]
        for response in self.responses:
            print("Job ARN", response["arn"])

    def stop_training(self):
        # # Cancelling robomaker job
        for job_arn in self.job_arns:
            self.robomaker.cancel_simulation_job(job=job_arn)

        # # Stopping sagemaker training job
        self.sage_session.sagemaker_client.stop_training_job(TrainingJobName=self.estimator._current_job_name)

    def clean_up_simualtion_environmnet(self):
        print('Cleaning Up Simulation Environment')
        self.robomaker.delete_simulation_application(application=self.simulation_app_arn)

    def plot_training_output(self):
        self.tmp_root = 'tmp/'
        os.system("mkdir {}".format(self.tmp_root))
        self.tmp_dir = "tmp/{}".format(self.job_name)
        os.system("mkdir {}".format(self.tmp_dir))
        print("Create local folder {}".format(self.tmp_dir))

        self.training_metrics_file = "training_metrics.json"
        self.training_metrics_path = "{}/{}".format(self.s3_prefix, self.training_metrics_file)

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
        #         ax = fig.add_subplot(1, 2, 1)

        x_axis = 'episode'
        y_axis = 'reward_score'
        ytwo_axis = 'completion_percentage'
        loops = self.job_duration_in_seconds
        x_entries = 0
        start_time = time.time()
        loops_without_change = 0
        with HiddenPrints():

            while self.check_if_simulation_active():

                wait_for_s3_object(self.s3_bucket, self.training_metrics_path, self.tmp_dir)

                json_file = "{}/{}".format(self.tmp_dir, self.training_metrics_file)
                with open(json_file) as fp:
                    data = json.load(fp)
                    data = pd.DataFrame(data['metrics'])

                    x = data[x_axis].values
                    y = data[y_axis].values
                    y2 = data[ytwo_axis].values
                    time_diff = time.time() - start_time
                    ax[0].title.set_text('Reward / Episode @ {} Seconds.'.format(time_diff))
                    ax[0].plot(x, y)
                    ax[1].title.set_text('Track Completion / Episode @ {} Seconds.'.format(time_diff))
                    ax[1].plot(x, y2)
                    fig.tight_layout()
                    display(fig)
                    clear_output(wait=True)
                    plt.pause(5)
#                 if x_entries == len(x):
#                     loops_without_change += 1
#                 else:
#                     x_entries = len(x)
#                 # finally break the loop and finish displaying...
#                 if loops_without_change > 5:
#                     clear_output(wait=True)
#                     display(fig)
#                     break
            clear_output(wait=True)
            display(fig)

    def start_evaluation_process(self):
        sys.path.append("./src")

        self.num_simulation_workers = 1

        self.evaluation_metrics_file = "{}/evaluation_metrics.json".format(self.s3_prefix)

        self.eval_envriron_vars = {
            "WORLD_NAME": self.track_name,
            "KINESIS_VIDEO_STREAM_NAME": "SilverstoneStream",
            "MODEL_S3_BUCKET": self.s3_bucket,
            "MODEL_S3_PREFIX": self.s3_prefix,
            "APP_REGION": self.aws_region,
            "MODEL_METADATA_FILE_S3_KEY": "%s/model_metadata.json" % self.s3_prefix,
            "METRICS_S3_BUCKET": self.s3_bucket,
            "METRICS_S3_OBJECT_KEY": self.evaluation_metrics_file,
            "NUMBER_OF_TRIALS": str(self.evaluation_trials),
            "ROBOMAKER_SIMULATION_JOB_ACCOUNT_ID": self.account_id
        }

        self.eval_simulation_application = {
            "application": self.simulation_app_arn,
            "launchConfig": {
                "packageName": "deepracer_simulation_environment",
                "launchFile": "evaluation.launch",
                "environmentVariables": self.eval_envriron_vars
            }
        }

        self.eval_vpcConfig = {"subnets": self.deepracer_subnets,
                               "securityGroups": self.deepracer_security_groups,
                               "assignPublicIp": True}

        responses = []
        for job_no in range(self.num_simulation_workers):
            response = self.robomaker.create_simulation_job(clientRequestToken=strftime("%Y-%m-%d-%H-%M-%S", gmtime()),
                                                            outputLocation={
                                                                "s3Bucket": self.s3_bucket,
                                                                "s3Prefix": self.s3_prefix
                                                            },
                                                            maxJobDurationInSeconds=self.job_duration_in_seconds,
                                                            iamRole=self.sagemaker_role,
                                                            failureBehavior="Continue",
                                                            simulationApplications=[self.eval_simulation_application],
                                                            vpcConfig=self.eval_vpcConfig)
            responses.append(response)

        # print("Created the following jobs:")
        for response in responses:
            print("Job ARN", response["arn"])

    def plot_evaluation_output(self):

        evaluation_metrics_file = "evaluation_metrics.json"
        evaluation_metrics_path = "{}/{}".format(self.s3_prefix, evaluation_metrics_file)
        wait_for_s3_object(self.s3_bucket, evaluation_metrics_path, self.tmp_dir)

        json_file = "{}/{}".format(self.tmp_dir, evaluation_metrics_file)
        with open(json_file) as fp:
            data = json.load(fp)

        df = pd.DataFrame(data['metrics'])
        # Converting milliseconds to seconds
        df['elapsed_time'] = df['elapsed_time_in_milliseconds'] / 1000
        df = df[['trial', 'completion_percentage', 'elapsed_time']]

        display(df)

    ### Starting here all methods related to multi-model training
    def param_gen_batch_sizes(self, min_batch=64, max_batch=512,
                              job_duration=3600,
                              job_name_prefix=None,
                              track_name='reinvent_base'):

        if job_name_prefix:
            batches = []
            btch = min_batch
            while btch <= max_batch:
                batches.append(btch)
                btch *= 2
            print(batches)

            model_params = []
            job_name = job_name_prefix + '-batchsize-'
            for batch_size in batches:
                params = {
                    'job_name': job_name + '{}'.format(batch_size),
                    'track_name': track_name,
                    'job_duration': job_duration,
                    'batch_size': batch_size,
                    'evaluation_trials': 5
                }
                model_params.append(params)
            print('{} Hyperparameter configs generated'.format(len(model_params)))

            return model_params

    ### Starting here all methods related to multi-model training
    def param_gen_tracks(self, job_name_prefix=None,
                         batch_size=64,
                         job_duration=3600,
                         track_names=['reinvent_base', 'Oval_track', 'Bowtie_track']):

        if job_name_prefix:

            print(track_names)

            model_params = []
            job_name = job_name_prefix + '-track-'
            for track in track_names:
                params = {
                    'job_name': job_name + '{}'.format(track.replace('_', '-')),
                    'track_name': track,
                    'job_duration': job_duration,
                    'batch_size': batch_size,
                    'evaluation_trials': 5
                }
                model_params.append(params)

            print('{} Hyperparameter configs generated'.format(len(model_params)))

            return model_params
        else:
            raise Exception('A Job Name Prefix needs to be specified. Exiting')

    def start_multi_model_simulations(self, params):

        drs = {}
        for param in params:
            # let's create a DeepRacerEngine instance and kick things off
            dr = DeepRacerEngine(param)
            dr.start_training_testing_process()
            drs[param['job_name']] = dr
            time.sleep(5)

        return drs

    def start_multi_model_evaluation(self, drs):

        for k, dr in drs.items():
            # Kickoff the evaluation!
            print('Kicking off RoboMaker Simulation for Job: {}.'.format(dr.job_name))
            dr.start_evaluation_process()
            time.sleep(5)

    def plot_multi_model_runs_output(self, drs):

        # create root for testing
        tmp_root = 'tmp/'
        os.system("mkdir {}".format(tmp_root))

        job_names = []
        tmp_dirs = []
        lngest_job_duration = 0
        for k, dr in drs.items():
            dr.tmp_dir = "tmp/{}".format(dr.job_name)
            os.system("mkdir {}".format(dr.tmp_dir))
            print("Create local folder {}".format(dr.tmp_dir))

            if dr.job_duration_in_seconds > lngest_job_duration:
                lngest_job_duration = dr.job_duration_in_seconds

            dr.training_metrics_file = "training_metrics.json"
            dr.training_metrics_path = "{}/{}".format(dr.s3_prefix, dr.training_metrics_file)

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
        #         ax = fig.add_subplot(1, 2, 1)

        x_axis = 'episode'
        y_axis = 'reward_score'
        ytwo_axis = 'completion_percentage'

        loops = lngest_job_duration
        x_entries = 0
        start_time = time.time()
        loops_without_change = 0
        with HiddenPrints():

            while self.check_if_simulation_active():
                try:
                    ax[0].legend_ = None
                except:
                    pass
                xs = {}
                ys = {}
                y2s = {}
                for k, dr in drs.items():
                    wait_for_s3_object(dr.s3_bucket, dr.training_metrics_path, dr.tmp_dir)

                    json_file = "{}/{}".format(dr.tmp_dir, dr.training_metrics_file)
                    with open(json_file) as fp:
                        data = json.load(fp)
                        data = pd.DataFrame(data['metrics'])
                        x = data[x_axis].values
                        y = data[y_axis].values
                        y2 = data[ytwo_axis].values
                        xs[k] = x
                        ys[k] = y
                        y2s[k] = y2

                time_diff = time.time() - start_time

                # now plot them all together
                for k, x in xs.items():
                    ax[0].title.set_text('Reward / Episode @ {} Seconds.'.format(time_diff))
                    ax[0].plot(x, ys[k], label=k)
                    ax[1].title.set_text('Track Completion / Episode @ {} Seconds.'.format(time_diff))
                    ax[1].plot(x, y2s[k], label=k)
                #                     ax[0].legend(loc="upper left")

                fig.tight_layout()
                display(fig)
                clear_output(wait=True)
                plt.pause(5)

#                 if x_entries == len(x):
#                     loops_without_change += 1
#                 else:
#                     x_entries = len(x)
#                 # finally break the loop and finish displaying...
#                 if loops_without_change > 20:
            clear_output(wait=True)
            display(fig)
                    

    def plot_multi_model_evaluation(self, drs):

        dfs = []
        for k, dr in drs.items():
            evaluation_metrics_file = "evaluation_metrics.json"
            evaluation_metrics_path = "{}/{}".format(dr.s3_prefix, evaluation_metrics_file)
            wait_for_s3_object(dr.s3_bucket, evaluation_metrics_path, dr.tmp_dir)

            json_file = "{}/{}".format(dr.tmp_dir, evaluation_metrics_file)
            with open(json_file) as fp:
                data = json.load(fp)

            df = pd.DataFrame(data['metrics'])
            # Converting milliseconds to seconds
            df['elapsed_time'] = df['elapsed_time_in_milliseconds'] / 1000
            df['job'] = dr.job_name
            df = df[['job', 'trial', 'completion_percentage', 'elapsed_time']]
            dfs.append(df)

        df = pd.concat(dfs)
        display(df)

    # Misc Functions
    def check_if_simulation_active(self):

        sims = self.robomaker.list_simulation_jobs()
        for sim in sims['simulationJobSummaries']:
            if sim['arn'] in self.job_arns:
                if sim['status'] == 'Running':
#                     print(sim)
                    return True
                else:
                    return False

    
    def delete_all_simulations(self):

        sims = self.robomaker.list_simulation_applications()
        for sim in sims['simulationApplicationSummaries']:
            self.robomaker.delete_simulation_application(application=sim['arn'])
            print('deleted ', sim['simulationApplicationNames'])

    def delete_s3_simulation_resources(self):

        # to-do
        print('Deleted s3 resources related to the {} Job.'.format(self.job_name))
        bucket = self.s3.Bucket(self.s3_bucket)
        bucket.objects.filter(Prefix=s3_prefix).delete()

  

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout














