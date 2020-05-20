import time
from enum import Enum
from io import StringIO
import re
import sys

import sagemaker
import boto3
import json
import logging
from threading import Thread
from sagemaker.rl.estimator import RLEstimator
from sagemaker.local.local_session import LocalSession
from sagemaker.analytics import TrainingJobAnalytics
from botocore.exceptions import ClientError

from orchestrator.clients.ddb.model_db_client import ModelDbClient
from orchestrator.workflow.datatypes.model_record import ModelRecord
from orchestrator.exceptions.ddb_client_exceptions import RecordAlreadyExistsException
from orchestrator.exceptions.workflow_exceptions import UnhandledWorkflowException

from src.vw_utils import EVAL_CHANNEL

logger = logging.getLogger("orchestrator")
            
            
class CaptureStdout(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, type, value, traceback):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout
        
        # Capture the exception and don't throw it back for graceful exit.
        return True


class ModelManager():
    """A model entity with the given experiment. This class will handle
    the model creation, model training, model evaluation and model metadata
    management.
    """

    def __init__(
            self,
            model_db_client: ModelDbClient,
            experiment_id,
            model_id,
            image=None,
            role=None,
            instance_config={},
            boto_session=None,
            algor_config={},
            train_state=None,
            evaluation_job_name=None,
            eval_state=None,
            eval_scores={},
            input_model_id=None,
            rl_estimator=None,
            input_data_s3_prefix=None,
            manifest_file_path=None,
            eval_data_s3_path=None,
            s3_model_output_path=None,
            training_start_time=None,
            training_end_time=None):
        """Initialize a model entity in the current experiment

        Args:
            model_db_client (ModelDBClient): A DynamoDB client
                to query the model table. The 'Model' entity use this client
                to read/update the model state.
            experiment_id (str): A unique id for the experiment. The created/loaded
                model will be associated with the given experiment.
            model_id (str): Aa unique id for the model. The model table uses
                model id to manage associated model metadata.
            image (str): The container image to use for training/evaluation.
            role (str): An AWS IAM role (either name or full ARN). The Amazon
                SageMaker training jobs will use this role to access AWS resources.
            instance_config (dict): A dictionary that specify the resource
                configuration for the model training/evaluation job.
            boto_session (boto3.session.Session): A session stores configuration
                state and allows you to create service clients and resources.
            algor_config (dict): A dictionary that specify the algorithm type 
                and hyper parameters of the training/evaluation job.
            train_state (str): State of the model training job.
            evaluation_job_name (str): Job name for Latest Evaluation Job for this model
            eval_state (str): State of the model evaluation job.
            input_model_id (str): A unique model id to specify which model to use
                as a pre-trained model for the model training job.
            rl_estimator (sagemaker.rl.estimator.RLEstimator): A Sagemaker RLEstimator
                entity that handle Reinforcement Learning (RL) execution within
                a SageMaker Training Job.
            input_data_s3_prefix (str): Input data path for the data source of the
                model training job.
            s3_model_output_path (str): Output data path of model artifact for the 
                model training job.
            training_start_time (str): Starting timestamp of the model training job.
            training_end_time (str): Finished timestamp of the model training job.

        Returns:
            orchestrator.model_manager.ModelManager: A ``Model`` object associated
            with the given experiment.
        """

        self.model_db_client = model_db_client
        self.experiment_id = experiment_id
        self.model_id = model_id

        # Currently we are not storing image/role and other model params in ModelDb
        self.image = image
        self.role = role
        self.instance_config = instance_config
        self.algor_config = algor_config

        # load configs
        self.instance_type = self.instance_config.get("instance_type", "local")
        self.instance_count = self.instance_config.get("instance_count", 1)
        self.algor_params = self.algor_config.get("algorithms_parameters", {})

        # create a local ModelRecord object. 
        self.model_record = ModelRecord(
            experiment_id,
            model_id,
            train_state,
            evaluation_job_name,
            eval_state,
            eval_scores,
            input_model_id,
            input_data_s3_prefix,
            manifest_file_path,
            eval_data_s3_path,
            s3_model_output_path,
            training_start_time,
            training_end_time
            )
        
        # try to save this record file. if it throws RecordAlreadyExistsException 
        # reload the record from ModelDb, and recreate
        try:
            self.model_db_client.create_new_model_record(
                self.model_record.to_ddb_record()
            )
        except RecordAlreadyExistsException:
            logger.debug("Model already exists. Reloading from model record.")
            model_record = self.model_db_client.get_model_record(
                experiment_id,
                model_id
            )
            self.model_record = ModelRecord.load_from_ddb_record(model_record)
        except Exception as e:
            logger.error("Unhandled Exception! " + str(e))
            raise UnhandledWorkflowException("Something went wrong while creating a new model")

        if boto_session is None:
            boto_session = boto3.Session()
        self.boto_session = boto_session

        if self.instance_type == 'local':
            self.sagemaker_session = LocalSession()
        else:
            self.sagemaker_session = sagemaker.session.Session(self.boto_session)
        self.sagemaker_client = self.sagemaker_session.sagemaker_client

    def _jsonify(self):
        """Return a JSON Dict with metadata of the ModelManager Object stored in
        self.model_record
        """
        return self.model_record.to_ddb_record()

    @classmethod
    def name_next_model(cls, experiment_id):
        """Generate unique model id of a new model in the experiment

        Args:
            experiment_id (str): A unique id for the experiment. The created/loaded
                model will be associated with the given experiment.

        Returns:
            str: A unique id for a new model
        """
        return experiment_id + "-model-id-" + str(int(time.time()))

    def _get_rl_estimator_args(self, eval=False):
        """Get required args to be used by RLEstimator class

        Args:
            eval (boolean): Boolean value to tell if the estimator is
                running a training/evaluation job.

        Return:
            dict: RLEstimator args used to trigger a SageMaker training job
        """
        entry_point = "eval-cfa-vw.py" if eval else "train-vw.py"
        estimator_type = "Evaluation" if eval else "Training"
        job_types = "evaluation_jobs" if eval else "training_jobs"

        sagemaker_bucket = self.sagemaker_session.default_bucket()
        output_path = f"s3://{sagemaker_bucket}/{self.experiment_id}/{job_types}/" 

        metric_definitions = [
            {
                'Name': 'average_loss',
                'Regex': 'average loss = ([-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?).*$'
            }
        ]

        args = dict(entry_point=entry_point,
                    source_dir='src',
                    dependencies=["common/sagemaker_rl"],
                    image_name=self.image,
                    role=self.role,
                    sagemaker_session=self.sagemaker_session,
                    train_instance_type=self.instance_type,
                    train_instance_count=self.instance_count,
                    metric_definitions=metric_definitions,
                    hyperparameters=self.algor_params,
                    output_path=output_path,
                    code_location=output_path.strip('/')
                    )

        if self.instance_type == 'local':
            logger.info(f"{estimator_type} job will be executed in 'local' mode")
        else:
            logger.info(f"{estimator_type} job will be executed in 'SageMaker' mode")
        return args

    def _fit_first_model(self, input_data_s3_prefix=None, manifest_file_path=None, wait=False, logs=True):
        """
        A Estimator fit() call to initiate the first model of the experiment
        """
        rl_estimator_args = self._get_rl_estimator_args()
        self.rl_estimator = RLEstimator(**rl_estimator_args)

        if manifest_file_path:
            input_data = sagemaker.session.s3_input(
                s3_data=manifest_file_path,
                input_mode='File',
                s3_data_type='ManifestFile'
                )
            self.rl_estimator.fit(job_name=self.model_id, inputs=input_data, wait=wait, logs=logs)
        else:
            self.rl_estimator.fit(job_name=self.model_id, inputs=input_data_s3_prefix, wait=wait,logs=logs)

    def fit(
            self,
            input_model_id=None,
            input_data_s3_prefix=None,
            manifest_file_path=None,
            wait=False,
            logs=True
            ):
        """A Estimator fit() call to start a model training job.

        Args:
            input_model_id (str): Model id of model to used as pre-trained model of the training job
            input_data_s3_prefix (str): Defines the location of s3 data to train on.
            manifest_file_path (str): Manifest file used to provide training data.
            wait (bool): Whether the call should wait until the job completes. Only
                meaningful when running in SageMaker mode.
            logs (bool): Whether to show the logs produced by the job.
                Only meaningful when wait is True (default: True).
        """
        # update object var, to be reflected in DDb Record as well.
        self.model_record.add_new_training_job_info(
            input_model_id=input_model_id,
            input_data_s3_prefix=input_data_s3_prefix,
            manifest_file_path=manifest_file_path
        )
        self.model_db_client.update_model_record(self._jsonify())

        if input_model_id is None:
            self._fit_first_model(
                input_data_s3_prefix=input_data_s3_prefix, 
                manifest_file_path=manifest_file_path, 
                wait=wait, 
                logs=logs)
        else:
            # use 'input_model_id' as pretrained model for training
            input_model_record = self.model_db_client.get_model_record(
                self.experiment_id,
                input_model_id
            )
            model_artifact_path = input_model_record.get("s3_model_output_path")
            rl_estimator_args = self._get_rl_estimator_args()
            rl_estimator_args['model_channel_name'] = 'pretrained_model'
            rl_estimator_args['model_uri'] = model_artifact_path
            self.rl_estimator = RLEstimator(**rl_estimator_args)

            if manifest_file_path:
                inputs = sagemaker.session.s3_input(
                    s3_data=manifest_file_path, 
                    s3_data_type='ManifestFile'
                )
            else:
                inputs = input_data_s3_prefix

            self.rl_estimator.fit(
                job_name=self.model_id,
                inputs=inputs,
                wait=wait,
                logs=logs
            )

    def evaluate(
            self,
            input_data_s3_prefix=None,
            manifest_file_path=None,
            evaluation_job_name=None,
            local_mode=True,
            wait=False,
            logs=True
            ):
        """A Estimator fit() call to start a model evaluation job.

        Args:
            input_data_s3_prefix (str): Defines the location of s3 data used for evaluation
            manifest_file_path (str): Manifest file used to provide evaluation data.
            evaluation_job_name (str): Unique Sagemaker job name to identify the evaluation job
            local_mode (bool): Whether the evaluation job is running on local mode
            wait (bool): Whether the call should wait until the job completes. Only
                meaningful when running in SageMaker mode.
            logs (bool): Whether to show the logs produced by the job.
                Only meaningful when wait is True.
        """
        # use self.model_id, self._s3_model_output_path as the model to evaluate
        # Model object has already been initialized with up-to-date DDb record.
        model_artifact_path = self.model_record.get_model_artifact_path()
        rl_estimator_args = self._get_rl_estimator_args(eval=True)
        rl_estimator_args['model_channel_name'] = 'pretrained_model'
        rl_estimator_args['model_uri'] = model_artifact_path

        if manifest_file_path:
            inputs = sagemaker.session.s3_input(
                s3_data=manifest_file_path, 
                s3_data_type='ManifestFile'
            )
            if local_mode:
                rl_estimator_args["hyperparameters"].update({"local_mode_manifest": True})

        else:
            inputs = input_data_s3_prefix
        
        # (dict[str, str] or dict[str, sagemaker.session.s3_input]) for evaluation channel
        eval_channel_inputs = {EVAL_CHANNEL: inputs}
        self.rl_estimator = RLEstimator(**rl_estimator_args)

        # update to save eval_data_s3_path in DDb as well, or 
        # update to read from SM describe call... maybe will not work in local mode but.
        eval_data_s3_path = manifest_file_path if (manifest_file_path is not None) else input_data_s3_prefix

        # we keep eval job state as pending, before the SM job has been submitted.
        # the syncer function should update this state, based on SM job status.
        self.model_record.add_new_evaluation_job_info(
            evaluation_job_name=evaluation_job_name,
            eval_data_s3_path=eval_data_s3_path
        )
        self.model_db_client.update_model_record(self._jsonify())

        # The following local variables (unsaved to DDb) make evaluation job non-resumable.
        self.log_output = None
        self.local_mode = local_mode

        if local_mode:
            # Capture eval score by regex expression
            # log should contain only one "average loss = some number" pattern
            with CaptureStdout() as log_output:
                self.rl_estimator.fit(
                            job_name=evaluation_job_name,
                            inputs=eval_channel_inputs,
                            wait=wait,
                            logs=logs
                        )

            self.log_output = '\n'.join(log_output)
            logger.debug(self.log_output)
        else:
            self.rl_estimator.fit(
                job_name=evaluation_job_name,
                inputs=eval_channel_inputs,
                wait=wait,
                logs=logs
            )

    def update_model_training_state(self):
        self._update_model_table_training_states()
    
    def update_model_evaluation_state(self):
        self._update_model_table_evaluation_states()

    def _update_model_table_training_states(self):
        """
        Update the training states in the model table. This method
        will poll the Sagemaker training job and then update
        training job metadata of the model, including:
            train_state,
            s3_model_output_path,
            training_start_time,
            training_end_time

        Args:
            model_record (dict): Current model record in the
                model table
        """
        if self.model_record.model_in_terminal_state():
            # model already in one of the final states
            # need not do anything.
            self.model_db_client.update_model_record(self._jsonify())
            return self._jsonify()
    
        # Else, try and fetch updated SageMaker TrainingJob status
        sm_job_info = {}
        for i in range(3):
            try:
                sm_job_info = self.sagemaker_client.describe_training_job(
                    TrainingJobName=self.model_id)
            except Exception as e:
                if "ValidationException" in str(e):
                    if i >= 2:
                        # 3rd attempt for DescribeTrainingJob failed with ValidationException
                        logger.warn(f"Looks like SageMaker Job was not submitted successfully."
                                    f" Failing Training Job with ModelId {self.model_id}"
                        )
                        self.model_record.update_model_as_failed()
                        self.model_db_client.update_model_as_failed(self._jsonify())
                        return
                    else: 
                        time.sleep(5)
                        continue
                else:
                    # Do not raise exception, most probably throttling. 
                    logger.warn(f"Failed to check SageMaker Training Job state for ModelId {self.model_id}."
                                " This exception will be ignored, and retried."
                    )
                    logger.debug(e)
                    time.sleep(2)
                    return self._jsonify()

        train_state = sm_job_info.get('TrainingJobStatus', "Pending")
        training_start_time = sm_job_info.get('TrainingStartTime', None)
        training_end_time = sm_job_info.get("TrainingEndTime", None)

        if training_start_time is not None:
            training_start_time =  training_start_time.strftime("%Y-%m-%d %H:%M:%S")
        if training_end_time is not None:
            training_end_time =  training_end_time.strftime("%Y-%m-%d %H:%M:%S")
        

        model_artifacts = sm_job_info.get('ModelArtifacts', None)
        if model_artifacts is not None:
            s3_model_output_path = model_artifacts.get("S3ModelArtifacts", None)
        else:
            s3_model_output_path = None

        self.model_record.update_model_job_status(
            training_start_time,
            training_end_time,
            train_state,
            s3_model_output_path
        )

        self.model_db_client.update_model_job_state(self._jsonify())

    def _update_model_table_evaluation_states(self):
        """Update the evaluation states in the model table. This method
        will poll the Sagemaker evaluation job and then update
        evaluation job metadata of the model, including:
            eval_state,
            eval_scores

        Args:
            model_record (dict): Current model record in the
                model table
        """

        if self.model_record.eval_in_terminal_state():
            self.model_db_client.update_model_record(
                self._jsonify()
            )
            return self._jsonify()
    
        # Try and fetch updated SageMaker Training Job Status
        sm_eval_job_info = {}
        for i in range(3):
            try:
                sm_eval_job_info = self.sagemaker_client.describe_training_job(
                    TrainingJobName=self.model_record._evaluation_job_name)
            except Exception as e:
                if "ValidationException" in str(e):
                    if i >= 2:
                        # 3rd attempt for DescribeTrainingJob with validation failure
                        logger.warn("Looks like SageMaker Job was not submitted successfully."
                                    f" Failing EvaluationJob {self.model_record._evaluation_job_name}"
                        )
                        self.model_record.update_eval_job_as_failed()
                        self.model_db_client.update_model_eval_as_failed(self._jsonify())
                        return
                    else: 
                        time.sleep(5)
                        continue
                else:
                    # Do not raise exception, most probably throttling. 
                    logger.warn("Failed to check SageMaker Training Job state for EvaluationJob: "
                                f" {self.model_record._evaluation_job_name}. This exception will be ignored,"
                                " and retried."
                    )
                    time.sleep(2)
                    return self._jsonify()
        

        eval_state = sm_eval_job_info.get('TrainingJobStatus', 'Pending')
        if eval_state == 'Completed':
            eval_score = "n.a."

            if self.local_mode:
                rgx = re.compile('average loss = ([-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?).*$', re.M)
                eval_score_rgx = rgx.findall(self.log_output)
        
                if len(eval_score_rgx) == 0:
                    logger.warning("No eval score available from vw job log.")
                else:
                    eval_score = eval_score_rgx[0][0] # [('eval_score', '')]
            else:
                attempts = 0
                while eval_score == 'n.a.' and attempts < 4:
                    try:
                        metric_df = TrainingJobAnalytics(self.model_record._evaluation_job_name, ['average_loss']).dataframe()
                        eval_score = str(metric_df[metric_df['metric_name'] == 'average_loss']['value'][0])
                    except Exception:
                        # to avoid throttling
                        time.sleep(5)
                        continue
                    attempts += 1
            self.model_record._eval_state = eval_state
            self.model_record.add_model_eval_scores(eval_score)
            self.model_db_client.update_model_eval_job_state(self._jsonify())
        else:
            # update eval state via ddb client
            self.model_record.update_eval_job_state(eval_state)
            self.model_db_client.update_model_eval_job_state(self._jsonify())