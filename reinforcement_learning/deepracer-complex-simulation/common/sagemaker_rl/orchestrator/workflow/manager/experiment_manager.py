import json
import logging
import time
from datetime import datetime, timedelta
from enum import Enum
from threading import Thread, Event
from packaging import version

import boto3
import docker

import sagemaker

logging.basicConfig()
logger = logging.getLogger('orchestrator')
logger.setLevel(logging.INFO)

try:
    assert version.parse(sagemaker.__version__) >= version.parse("1.33.0"), "sagemaker version should be >= 1.33.0"
except Exception as e:
    logger.error(
        """Please update your SageMaker SDK Version.
        You can do so by running this command in your Notebook Cell.\n\n
        !pip install --upgrade sagemaker\n\n
        """
    )
    logger.warn("You may need to restart the notebook Kernel as well")
    raise e

from botocore.exceptions import ClientError
from sagemaker.local.local_session import LocalSession

from orchestrator.clients.ddb.join_db_client import JoinDbClient
from orchestrator.clients.ddb.model_db_client import ModelDbClient
from orchestrator.clients.ddb.experiment_db_client import ExperimentDbClient
from orchestrator.workflow.manager.join_manager import JoinManager
from orchestrator.workflow.manager.model_manager import ModelManager
from orchestrator.workflow.datatypes.experiment_record import ExperimentRecord
from orchestrator.resource_manager import Predictor
from orchestrator.resource_manager import ResourceManager
from orchestrator.utils.cloudwatch_logger import CloudWatchLogger
from orchestrator.exceptions.ddb_client_exceptions import RecordAlreadyExistsException
from orchestrator.exceptions.workflow_exceptions import UnhandledWorkflowException, \
    SageMakerHostingException, SageMakerTrainingJobException, WorkflowJoiningJobException, \
    EvalScoreNotAvailableException, InvalidUsageException




class HostingState(str, Enum):
    PENDING = "PENDING"     # A hosting update request is pending
    DEPLOYING = "DEPLOYING" # A hosting update request is in process
    DEPLOYED = "DEPLOYED"   # Hosting update request was completed.
    FAILED = "FAILED"       # hosting update request failed.


class TrainingState(str, Enum):
    PENDING = "PENDING"             # A new model/training job create request is made
    TRAINING = "TRAINING"           # Model/Training job is in status of 'Training'
    TRAINED = "TRAINED"             # Model/Training job has been completed
    STOPPED = "STOPPED"             # Model/Training job has been stopped
    FAILED = "FAILED"               # Model/Training job has been failed


class EvaluationState(str, Enum):
    PENDING = "PENDING"             # A new evaluation job create request is made
    EVALUATING = "EVALUATING"       # Evaluation job is in status of 'Evaluating'
    EVALUATED = "EVALUATED"         # Evaluation job has been completed
    STOPPED = "STOPPED"             # Evaluation job has been stopped
    FAILED = "FAILED"               # Evaluation job has been failed


class JoiningState(str, Enum):
    PENDING = "PENDING"     # A joining request is pending
    RUNNING = "RUNNING"     # A joining job is running
    SUCCEEDED = "SUCCEEDED" # A joining job has been completed
    FAILED = "FAILED"       # A joining job has been failed
    CANCELLED = "CANCELLED" # A joining job has been cancelled

# Using SageMakerTrainingJob primary status 
TRAINING_JOB_STATUS_MAP = {
    "Pending": TrainingState.PENDING,
    "InProgress": TrainingState.TRAINING,
    "Stopping": TrainingState.TRAINING,
    "Stopped": TrainingState.STOPPED,
    "Failed": TrainingState.FAILED,
    "Completed": TrainingState.TRAINED
}

# Using SageMakerTrainingJob primary status 
EVALUATION_JOB_STATUS_MAP = {
    "Pending": EvaluationState.PENDING,
    "InProgress": EvaluationState.EVALUATING,
    "Stopping": EvaluationState.EVALUATING,
    "Stopped": EvaluationState.STOPPED,
    "Failed": EvaluationState.FAILED,
    "Completed": EvaluationState.EVALUATED
}

# Using SageMakerHostingEndpoint primary status 
HOSTING_ENDPOINT_STATUS_MAP = {
    "OutOfService": HostingState.FAILED,
    "Creating": HostingState.DEPLOYING,
    "Updating": HostingState.DEPLOYING,
    "SystemUpdating": HostingState.DEPLOYING,
    "RollingBack": HostingState.DEPLOYING,
    "InService": HostingState.DEPLOYED,
    "Deleting": HostingState.DEPLOYING,
    "Failed": HostingState.FAILED
}


class ExperimentManagerSyncThread(Thread):
    """A thread to synchronize states of the experiment to experiment table.
    The thread will keep running if running in non-local mode. The thread will
    first load the latest state from ddb table to local, check if there is
    any 'ongoing' state of the workflow. If it is, check the related table
    for the latest state and update the table.
    """

    def __init__(
        self,
        experiment_manager
    ):
        """Initialize a synchronization thread for the experiment

        Args:
            experiment_manager (ExperimentManager): ExperimentManager object
                with associated states
        """
        Thread.__init__(self)

        self.experiment_manager = experiment_manager
        self.experiment_id = experiment_manager.experiment_id

        self.exp_db_client = experiment_manager.exp_db_client
        self.model_db_client = experiment_manager.model_db_client
        self.join_db_client = experiment_manager.join_db_client
        self.sagemaker_client = experiment_manager.sagemaker_client

        # used to emit continuous CW Metrics (for Number type plot)
        self.latest_trained_model_id = None
        self.latest_trained_model_eval_score = None
        self.latest_hosted_model_id = None
        self.latest_hosted_model_eval_score = None

        self.thread_running = Event()
        self.thread_running.set()

    def _update_experiment_db_training_workflow_metadata(self, training_workflow_metadata):
        """
        Three thing happens here:
        a) Checks if current TrainingWorkflowMetadata needs an update.
        b) Fetches latest TrainingJob state from ModelDb for next_model_to_train
        c) Updates ExperimentDb TrainingWorkflowMetadata with latest information. 
        d) Finally, updates the local ExperimentManager context to latest.
        
        Args:
            training_workflow_metadata (dict): A dictionary containing
                training workflow related metadata
        """
        if training_workflow_metadata is None:
            # A training request hasn't been made yet. 
            # Nothing to proccess. Return.
            return
        
        next_model_to_train_id = training_workflow_metadata.get("next_model_to_train_id", None)
        training_state = training_workflow_metadata.get("training_state", None)

        if training_state is None:
            # A training request hasn't been made yet. 
            # Nothing to proccess. Return.
            return
        elif not training_state.endswith("ING"):
            # TrainingWorkflowState in ExperimentDb is already in a Final State.
            # Sync thread only updates on Pending/Running TrainingWorkflowState state.
            return
        elif training_state.endswith("ING") and next_model_to_train_id is None:
            # A training is in progress, but the training model-id is None!
            logger.warn(f"Model Training in {training_state}, while next_model_to_train_id is None. "
                         "Training Workflow would be stuck if this continues."
            )
            return
        else:
            # A training is in progress. Fetch the status of that training job from ModelDb.
            training_job_record = self.model_db_client.get_model_record_with_retry(
                self.experiment_id, next_model_to_train_id)

            # Get updated TrainingWorkflowState in {new_training_state}
            if training_job_record is None:
                # Training Job Record not found in ModelDb even with 1 retry, after 5 seconds.
                # Most likely there was a failure creating requested TrainingJob
                # Update the TrainingWorkflowState to Failed.
                logger.warn(f"ModelId {next_model_to_train_id} record not found. Failing the TrainingWorkflow")
                new_training_state = TrainingState.FAILED
            else:
                train_state_from_modeldb = training_job_record.get("train_state")

                if train_state_from_modeldb is not None:
                    new_training_state = TRAINING_JOB_STATUS_MAP[train_state_from_modeldb]
                else:
                    # Since ModelDb training job state is None, 
                    # keep the ExperimentDb TrainingWorkflowState same.
                    logger.warn(f"ModelDb has model-id {next_model_to_train_id} 's state as 'None'. "
                                "Training Worklow would be stuck if this continues."
                    )
                    new_training_state = training_state

        expected_next_model_to_train_id = next_model_to_train_id
        # Generate new TrainingWorkflowState for ExperimentDb based on new_training_state
        if new_training_state == TrainingState.TRAINED:
            training_workflow_metadata['last_trained_model_id'] = next_model_to_train_id
            training_workflow_metadata['next_model_to_train_id'] = None
            training_workflow_metadata['training_state'] = new_training_state

        elif new_training_state == TrainingState.FAILED or new_training_state == TrainingState.STOPPED:
            # training_workflow_metadata['last_trained_model_id'] remains the same
            # training_workflow_metadata['next_model_to_train_id'] remains the same or change to None
            # update the ExperimentDb TrainingWorkflowState to Failed
            training_workflow_metadata['training_state'] = new_training_state
        else:
            # training_workflow_metadata['last_trained_model_id'] remains the same
            # training_workflow_metadata['next_model_to_train_id'] remains the same
            # update the ExperimentDb TrainingWorkflowState to new_training_state
            training_workflow_metadata['training_state'] = new_training_state

        # Try to save the update in ExperimentDb
        # This can update the status only if in the current record,
        # next_model_to_train_id == expected_next_model_to_train_id
        try:
            self.exp_db_client.update_training_workflow_metadata_with_validation(
                self.experiment_id,
                training_workflow_metadata,
                expected_next_model_to_train_id
            )
        except Exception as e:
            if "ConditionalCheckFailedException" in str(e):
                # Most likely Sync Thread went out of sync :( 
                # Just return here without updating local ExperimentManager. 
                logger.warn("Sync Thread trying to update ExperimentDb with old state. This should "
                            "get fixed in next run!"
                )
                return
            logger.error("Failed to update ExperimentDb with latest information: " + str(e))
            raise UnhandledWorkflowException("Some error occurred while update ExperimentDb record TrainingWorkflowMetadata")

        # Finally, update local ExperimentManager with new states.
        self.experiment_manager.experiment_record._last_trained_model_id = training_workflow_metadata['last_trained_model_id']
        self.experiment_manager.experiment_record._next_model_to_train_id = training_workflow_metadata['next_model_to_train_id']
        self.experiment_manager.experiment_record._training_state = training_workflow_metadata['training_state']
            

    def _update_experiment_db_evaluation_workflow_metadata(self, evaluation_workflow_metadata):
        """
        Update the evaluation workflow metadata in the experiment table

        Args:
            evaluation_workflow_metadata (dict): A dictionary containing
                evaluation workflow related metadata
        """
        if evaluation_workflow_metadata is None:
            return

        evaluation_state = evaluation_workflow_metadata.get("evaluation_state", None)
        next_evaluation_job_id = evaluation_workflow_metadata.get("next_evaluation_job_id", None)

        # some evaluation request is in progress
        if evaluation_state is not None and evaluation_state.endswith("ING"):
            evaluation_model_id = next_evaluation_job_id.split('-eval-')[0]
            evaluation_job_record = self.model_db_client.get_model_record(
                self.experiment_id, evaluation_model_id)

            # if evaluation model record exists in the model table
            if evaluation_job_record is not None:

                eval_state_from_modeldb = evaluation_job_record.get("eval_state", None)

                # avoid overwrite joining_state into None
                if eval_state_from_modeldb is not None:
                    evaluation_state = EVALUATION_JOB_STATUS_MAP[eval_state_from_modeldb]

                self.experiment_manager.experiment_record._evaluation_state = evaluation_state
                # update table states via ddb client
                self.exp_db_client.update_experiment_evaluation_state(
                    self.experiment_id, evaluation_state
                )

                if evaluation_state == EvaluationState.EVALUATED:
                    self.experiment_manager.experiment_record._last_evaluation_job_id = next_evaluation_job_id
                    self.experiment_manager.experiment_record._next_evaluation_job_id = None

                    self.exp_db_client.update_experiment_last_evaluation_job_id(
                        self.experiment_id, next_evaluation_job_id
                    )
                    self.exp_db_client.update_experiment_next_evaluation_job_id(
                        self.experiment_id, None
                    )
                    
                    # update latest_train/eval metrics to publish to CW
                    self._update_metrics_from_latest_eval_job(next_evaluation_job_id)

    def _update_experiment_db_hosting_workflow_metadata(self, hosting_workflow_metadata):
        """Update the hosting workflow metadata in the experiment table
        
        Args:
            hosting_workflow_metadata (dict): A dictionary containing
                hosting workflow related metadata
        """        
        if hosting_workflow_metadata is None:
            return

        hosting_state = hosting_workflow_metadata.get("hosting_state", None)
        hosting_endpoint = hosting_workflow_metadata.get("hosting_endpoint", None)
        next_model_to_host_id = hosting_workflow_metadata.get("next_model_to_host_id", None)
        last_hosted_model_id = hosting_workflow_metadata.get("last_hosted_model_id", None)

        # confirm if deploy status is correct by sending a request.
        if hosting_state == HostingState.DEPLOYED and last_hosted_model_id:
            try:
                predictor = self.experiment_manager.predictor
                model_id = predictor.get_hosted_model_id()
                assert model_id == last_hosted_model_id
            except Exception:
                self.exp_db_client.update_experiment_hosting_state(
                    self.experiment_id, None
                )
                self.exp_db_client.update_experiment_hosting_endpoint(
                    self.experiment_id, None
                )
                self.experiment_manager.experiment_record._hosting_state = None
                self.experiment_manager.experiment_record._hosting_endpoint = None

        # some deployment request is in progress
        if hosting_state is not None and hosting_state.endswith("ING"):
            if (hosting_endpoint is None) or (not self.experiment_manager.soft_deployment):
                # deployment happen with a new endpoint initiation or blue green deployment
                # describe endpoint to get state of the deployment

                try:
                    sm_endpoint_info = self.sagemaker_client.describe_endpoint(
                        EndpointName=self.experiment_id
                    )
                except Exception:
                    # Do not raise exception
                    return

                hosting_state = HOSTING_ENDPOINT_STATUS_MAP[sm_endpoint_info.get("EndpointStatus")]

                self.experiment_manager.experiment_record._hosting_state = hosting_state
                # update table states via ddb client
                self.exp_db_client.update_experiment_hosting_state(
                    self.experiment_id, hosting_state
                )

                if hosting_state == HostingState.DEPLOYED:
                    # update local record
                    self.experiment_manager.experiment_record._hosting_endpoint = sm_endpoint_info.get("EndpointArn")
                    self.experiment_manager.experiment_record._last_hosted_model_id = next_model_to_host_id
                    self.experiment_manager.experiment_record._next_model_to_host_id = None
                    
                    # update DynamoDB record
                    self.exp_db_client.update_experiment_hosting_endpoint(
                        self.experiment_id, sm_endpoint_info.get("EndpointArn")
                    )
                    self.exp_db_client.update_experiment_last_hosted_model_id(
                        self.experiment_id, next_model_to_host_id
                    )
                    self.exp_db_client.update_experiment_next_model_to_host_id(
                        self.experiment_id, None
                    )

                    self._update_metrics_from_latest_hosting_update(next_model_to_host_id)
            else:
                # deployment happened on existing endpoint
                if self.experiment_manager.soft_deployment:
                    # query endpoint to get the current hosted model id

                    model_id = ""
                    num_retries = 0
                    while model_id != next_model_to_host_id:
                        predictor = self.experiment_manager.predictor
                        model_id = predictor.get_hosted_model_id()
                        num_retries += 1
                        if (not self.experiment_manager.local_mode) or (num_retries >= 5):
                            break
                        time.sleep(1)

                    # hosted model id got updated
                    if model_id == next_model_to_host_id:
                        hosting_state = HostingState.DEPLOYED
                    else:
                        hosting_state = HostingState.DEPLOYING

                    self.experiment_manager.experiment_record._hosting_state = hosting_state
                    # update hosting_state in exp table
                    self.exp_db_client.update_experiment_hosting_state(
                        self.experiment_id, hosting_state
                    )

                    if hosting_state == HostingState.DEPLOYED:
                        # update local record
                        self.experiment_manager.experiment_record._last_hosted_model_id = next_model_to_host_id
                        self.experiment_manager.experiment_record._next_model_to_host_id = None
                        
                        # update DynamoDB record
                        self.exp_db_client.update_experiment_last_hosted_model_id(
                            self.experiment_id, next_model_to_host_id
                        )
                        self.exp_db_client.update_experiment_next_model_to_host_id(
                            self.experiment_id, None
                        )
                        self._update_metrics_from_latest_hosting_update(next_model_to_host_id)

    def _update_experiment_db_joining_workflow_metadata(self, joining_workflow_metadata):
        """Update the joining workflow metadata in the experiment table
        
        Args:
            joining_workflow_metadata (dict): A dictionary containing
                joining workflow related metadata
        """          
        if joining_workflow_metadata is None:
            return

        joining_state = joining_workflow_metadata.get("joining_state", None)
        next_join_job_id = joining_workflow_metadata.get("next_join_job_id", None)

        # some joining job request is in progress
        if joining_state is not None and joining_state.endswith("ING"):
            join_job_record = self.join_db_client.get_join_job_record(
                self.experiment_id, next_join_job_id)

            # if join job record exists in the join table
            if join_job_record is not None:

                current_state = join_job_record.get("current_state", None)

                # avoid overwrite joining_state into None
                if current_state is not None:
                    joining_state = current_state

                self.experiment_manager.experiment_record._joining_state = joining_state
                # update table states via ddb client
                self.exp_db_client.update_experiment_joining_state(
                    self.experiment_id, joining_state
                )

                if joining_state == JoiningState.SUCCEEDED:
                    self.experiment_manager.experiment_record._last_joined_job_id = next_join_job_id
                    self.experiment_manager.experiment_record._next_join_job_id = None

                    self.exp_db_client.update_experiment_last_joined_job_id(
                        self.experiment_id, next_join_job_id
                    )
                    self.exp_db_client.update_experiment_next_join_job_id(
                        self.experiment_id, None
                    )

    def _update_metrics_from_latest_eval_job(self, latest_evaluation_job_id):
        """
        Updates SyncThread's local information on every Evaluation Job complete run.

        Also Emit CW metric for New Model Evaluation Scores plot, while updating 
        local latest_trained_model_* information, for continuous CW puts (for Number plots)
        """
        try:
            last_trained_model_id = self.experiment_manager.last_trained_model_id
            currently_hosted_model_id = self.experiment_manager.last_hosted_model_id
            
            if last_trained_model_id in latest_evaluation_job_id:
                # using in as latest_evaluation_job_id would be of format last_trained_model_id-{eval}-{timestamp}
                # If the EvaluationJob was for latest Trained Model 
                eval_score = self.get_latest_eval_score_for_model_id(last_trained_model_id)
                if eval_score == "n.a.":
                    logger.debug("EvalScore from last run in n.a.")
                    return
                else:
                    logger.debug("Updated Latest Trained Mode Eval Score")
                    self.latest_trained_model_id = last_trained_model_id
                    self.latest_trained_model_eval_score = eval_score

                    # Also publish this score once, for Eval Score over time Graph
                    self.experiment_manager.cw_logger.publish_newly_trained_model_eval_information(
                        self.experiment_id,
                        last_trained_model_id,
                        eval_score
                    )
            elif currently_hosted_model_id in latest_evaluation_job_id:
                # using in as latest_evaluation_job_id would be of format currently_hosted_model_id-{eval}-{timestamp}
                # If the EvaluationJob was for Currently Hosted Model
                eval_score = self.get_latest_eval_score_for_model_id(currently_hosted_model_id)
                if eval_score == "n.a.":
                    logger.debug("EvalScore for HostedModel is n.a.")
                    return
                else:
                    logger.debug("Updated Hosted Model Latest Eval score")
                    self.latest_hosted_model_eval_score = eval_score
            else:
                # Evaluation Job not for latest-trained-model
                logger.debug("Latest Evaluated Model doesn't match Latest Trained Model, or"
                             " Currently Hosted Model. Skipping reporting EvalScore")
                return

        except Exception as e:
            logger.warn("Failed to emit latest training job eval metrics." + str(e))

    def _update_metrics_from_latest_hosting_update(self, latest_hosted_model_id):
        """
        Updates SyncThread's local information on every Hosting Update completion
        """
        try:
            self.latest_hosted_model_id = latest_hosted_model_id
            eval_score = self.get_latest_eval_score_for_model_id(latest_hosted_model_id)
            if eval_score == "n.a.":
                logger.debug("EvalScore for latest hosted model is n.a.")
                return
            else:
                logger.debug("Updated Latest Eval Score")
                self.latest_trained_model_eval_score = eval_score

                # Also publish this score once, for Eval Score over time Graph
                self.experiment_manager.cw_logger.publish_latest_hosting_information(
                    self.experiment_id,
                    latest_hosted_model_id,
                    eval_score
                )
        except Exception as e:
            logger.warn("Failed to emit latest training job eval metrics." + str(e))

    def get_latest_eval_score_for_model_id(self, model_id):
        model_record = self.model_db_client.get_model_record(
            self.experiment_id,
            model_id
            )
        eval_score = "n.a."
        if model_record is not None:
            eval_keys = model_record["eval_scores"].keys()
            if eval_keys is None or len(eval_keys) == 0:
                # No EvalScore is available yet.
                return eval_score
            # sort eval score by s3 prefix as joining job is ordered by time
            eval_keys = sorted(eval_keys)
            return model_record["eval_scores"][eval_keys[-1]] 
        else: 
            return eval_score
    
    def emit_cloudwatch_metrics_for_training_and_hosting(self):
        try:
            # emit CloudWatch Training metrics
            if self.latest_trained_model_id and self.latest_trained_model_eval_score:
                self.experiment_manager.cw_logger.publish_latest_training_information(
                    self.experiment_id,
                    self.latest_trained_model_id,
                    self.latest_trained_model_eval_score
                )
            else:
                #logger.debug("Train CW Metrics Not Set")
                pass
        except Exception:
            logger.debug("Failed to publish CW Metrics for Training State")
            logger.debug(e)

        try:        
            # emit CloudWatch Hosting metrics
            if self.latest_hosted_model_id and self.latest_hosted_model_eval_score:
                self.experiment_manager.cw_logger.publish_latest_hosting_information(
                    self.experiment_id,
                    self.latest_hosted_model_id,
                    self.latest_hosted_model_eval_score
                )
            else:
                #logger.debug("Host CW Metrics Not Set")
                pass
        except Exception:
            logger.debug("Failed to publish CW Metrics for Training State")
            logger.debug(e)

    def sync_experiment_state_with_ddb(self):
        """
        Synchronize ExperimentDb states to local and update
        states of Training/Evaluation and Hosting workflows

        """
        record = self.exp_db_client.get_experiment_record(self.experiment_id)

        # sync records to experiment states
        self.experiment_manager.experiment_record = ExperimentRecord.load_from_ddb_record(record)

        # update training workflow if needed
        training_workflow_metadata = record.get("training_workflow_metadata", None)
        # first update any in-progress next_model_to_train
        next_model_to_train_id = self.experiment_manager.experiment_record._next_model_to_train_id
        training_state = self.experiment_manager.experiment_record._training_state
        if next_model_to_train_id is not None and training_state.endswith("ING"):
            if self.experiment_manager.next_model_to_train is not None:
                self.experiment_manager.next_model_to_train.update_model_training_state()
            else:
                # only init the ModelManager() if the training job record already exists
                if self.model_db_client.get_model_record(self.experiment_id, next_model_to_train_id) is not None:
                    next_model_to_train = ModelManager(
                        model_db_client=self.model_db_client,
                        experiment_id=self.experiment_id,
                        model_id=next_model_to_train_id)
                    next_model_to_train.update_model_training_state()
        time.sleep(1)
        self._update_experiment_db_training_workflow_metadata(training_workflow_metadata)

        # update evaluation workflow if needed
        evaluation_workflow_metadata = record.get("evaluation_workflow_metadata", None)
        # first update any in-progress next_evaluation_job
        next_evaluation_job_id = self.experiment_manager.experiment_record._next_evaluation_job_id
        evaluation_state = self.experiment_manager.experiment_record._evaluation_state
        if next_evaluation_job_id is not None and evaluation_state.endswith("ING"):
            if self.experiment_manager.next_model_to_evaluate is not None:
                self.experiment_manager.next_model_to_evaluate.update_model_evaluation_state()
            else:
                # only init the ModelManager() if the evaluation job record already exists
                if self.model_db_client.get_model_record(self.experiment_id, \
                    next_evaluation_job_id.split('-eval-')[0]) is not None:
                    next_model_to_evaluate = ModelManager(
                        model_db_client=self.model_db_client,
                        experiment_id=self.experiment_id,
                        model_id=next_evaluation_job_id.split('-eval-')[0])
                    next_model_to_evaluate.update_model_evaluation_state()
        time.sleep(1)
        self._update_experiment_db_evaluation_workflow_metadata(evaluation_workflow_metadata)

        # update hosting workflow if needed
        hosting_workflow_metadata = record.get("hosting_workflow_metadata", None)
        self._update_experiment_db_hosting_workflow_metadata(hosting_workflow_metadata)

        # update joining workflow if needed
        joining_workflow_metadata = record.get("joining_workflow_metadata", None)
        # first update any in-progress next_join_job
        next_join_job_id = self.experiment_manager.experiment_record._next_join_job_id
        joining_state = self.experiment_manager.experiment_record._joining_state
        if next_join_job_id is not None and joining_state.endswith("ING"):
            if self.experiment_manager.next_join_job is not None:
                self.experiment_manager.next_join_job.update_join_job_state()
            else:
                # only init the JoinManager() if the join job record already exists
                if self.join_db_client.get_join_job_record(self.experiment_id, next_join_job_id) is not None:
                    next_join_job = JoinManager(
                        join_db_client=self.join_db_client,
                        experiment_id=self.experiment_id,
                        join_job_id=next_join_job_id)
                    next_join_job.update_join_job_state()
        time.sleep(1)
        self._update_experiment_db_joining_workflow_metadata(joining_workflow_metadata)

        self.emit_cloudwatch_metrics_for_training_and_hosting()

    def run(self):
        """
        Start to run the daemon thread for states synchronization
        """
        logger.debug("Starting a daemon thread to sync experiment states")
        while self.thread_running.is_set():
            try:
                self.sync_experiment_state_with_ddb()
            except Exception as e:
                logger.warn("Exception occurred in Experiment Sync Thread: " + str(e))
                logger.error(e)
                logger.warn("Resuming Sync in 10 seconds...")
                time.sleep(10)
            time.sleep(.5)


class ExperimentManager():
    """
    A experiment entity to manage different components in the continual learning
    iteration loops. One experiment will be initiated to solve a single RL problem.
    An experiment can be created or loaded by an unique experiment id. The experiment
    entity provides methods/functionalities for model training/evaluation/deployment
    and data joining.
    """
    
    def __init__(self,
                 config,
                 experiment_id,
                 training_workflow_metadata={},
                 hosting_workflow_metadata={},
                 joining_workflow_metadata={},
                 evaluation_workflow_metadata={}
                 ):
        """Initialize/Reload an experiment entity to manage the workflow

        Args:
            config (dict): Config values for the experiment setup
            experiment_id (str): A unique experiment id for the experiment
            training_workflow_metadata (dict): Metadata for the training workflow
            hosting_workflow_metadata (dict): Metadata for the hosting workflow
            joining_workflow_metadata (dict): Metadata for the joining workflow
            evaluation_workflow_metadata (dict): Metadata for the evaluation workflow

        Return:
            sagemaker_rl.orchestrator.workflow.experiment_manager.ExperimentManager: A ``ExperimentManager`` object
            to manage the workflow
        """                 
        self.boto_session = boto3.Session()
        self._region_name = self.boto_session.region_name
        self.account = self.boto_session.client("sts").get_caller_identity()["Account"]
        if self._region_name is None:
            raise ValueError('Must setup AWS configuration with a valid region')

        # unique id common across all experiments in the account
        self.experiment_id = experiment_id

        # load configs
        self.config = config
        self.image = self.config.get("image", None).replace("{AWS_REGION}", self._region_name)
        self.algor_config = self.config.get("algor", {})
        self.local_mode = self.config.get("local_mode", True)
        if self.local_mode:
            self._update_instance_type_for_local_mode()
            self.sagemaker_session = LocalSession()
        else:
            self.sagemaker_session = sagemaker.session.Session(self.boto_session)

        self.soft_deployment = self.config.get("soft_deployment", False)

        # load resource config and init shared resourced if not exists
        self.resource_manager = ResourceManager(self.config.get("resource", {}),
                                                boto_session=self.boto_session)
        self.resource_manager.create_shared_resource_if_not_exist()

        # init clients
        self.exp_db_client = self.resource_manager.exp_db_client
        self.model_db_client = self.resource_manager.model_db_client
        self.join_db_client = self.resource_manager.join_db_client
        self.cw_logger = CloudWatchLogger(
            self.boto_session.client("cloudwatch"),
            self._region_name
            )
        self.sagemaker_client = self.sagemaker_session.sagemaker_client

        # init s3 client for rewards upload
        self.s3_client = self.boto_session.client('s3')

        # create a local JoinJobRecord object. 
        self.experiment_record = ExperimentRecord(
            experiment_id,
            training_workflow_metadata,
            hosting_workflow_metadata,
            joining_workflow_metadata,
            evaluation_workflow_metadata
        )
        self.next_model_to_train = None
        self.next_join_job = None
        self.next_model_to_evaluate = None

        # Try to save new ExperimentRecord to ExperimentDb. If it throws 
        # RecordAlreadyExistsException, re-read the ExperimentRecord from ExperimentDb,
        # and use it as initial state
        try:
            self.exp_db_client.create_new_experiment_record(
                self.experiment_record.to_ddb_record()
            )
        except RecordAlreadyExistsException:
            logger.warn(f"Experiment with name {self.experiment_id} already exists. "
                        "Reusing current state from ExperimentDb.")
            experiment_record = self.exp_db_client.get_experiment_record(
                experiment_id
            )
            self.experiment_record = ExperimentRecord.load_from_ddb_record(experiment_record)
        except Exception as e:
            logger.error("Unhandled Exception! " + str(e))
            raise UnhandledWorkflowException("Something went wrong while creating a new experiment")

        try:
            self.cw_logger.create_cloudwatch_dashboard_from_experiment_id(
                self.experiment_id
            )
        except Exception as e:
            logger.error("Unable to create CloudWatch Dashboard." + str(e))
            logger.error("To see metrics on CloudWatch, run bandit_experiment."
                         "cw_logger.create_cloudwatch_dashboard_from_experiment_id function again.")


        # start a daemon thread to sync ExperimentDb states to local states
        # the daemon thread will keep running till the session ends
        self.sync_thread = ExperimentManagerSyncThread(experiment_manager=self)

        # Run the thread in SageMaker mode only
        if not self.local_mode:
           self.sync_thread.setDaemon(True)
           self.sync_thread.start()

    def _sync_experiment_state_with_ddb(self):
        """
        Synchronize table states into the object states. This method only be
        invoked in local mode.
        """
        if self.local_mode:
            self.sync_thread.sync_experiment_state_with_ddb()

    def _update_instance_type_for_local_mode(self):
        """Update the instance type if running in 'local' mode
        """
        self.config["resource"]["private_resource"]["hosting_fleet"]["instance_type"] = "local"
        self.config["resource"]["private_resource"]["training_fleet"]["instance_type"] = "local"
        self.config["resource"]["private_resource"]["evaluation_fleet"]["instance_type"] = "local"

    def _jsonify(self):
        """Return a jsonify dict with metadata of the 'Experiment' object
        """        
        return self.experiment_record.to_ddb_record()

    def _get_prefix_and_relative_path(self, path_list):
        """Return shared prefix and relative paths given a list of paths
        
        Args:
            path_list (list): A list of string representing S3 paths
        
        Returns:
            (str, list): Return shared prefix and a list of relative paths
        """
        # example of path: s3://custom-bucket/exp-1/exp-1-join-id-time-stamp/train
        # use s3 bucket as prefix
        # allow data from different experiments but in same account
        parts = path_list[0].split('/')
        shared_prefix = '/'.join(parts[0:3])  # s3://custom-bucket
        key_path_list = []

        for path in path_list:
            parts = path.split('/')
            prefix = '/'.join(parts[0:3])
            if prefix != shared_prefix:
                logger.error(f" Prefix `{prefix}` is different from the shared prefix '{shared_prefix}'. "
                             "Data in the list are not coming from same s3 bucket.")
            object_path = '/'.join(parts[3:])
            key_path_list.append(object_path)

        return shared_prefix, key_path_list

    def _write_manifest_to_s3(self, manifest_file):
        """Upload manifest file to S3 bucket
        
        Args:
            manifest_file (dict): A json blob that contains manifest shared prefix
                and list of relative paths
        
        Returns:
            str: S3 data path for the uploaded manifest file
        """
        account = self.boto_session.client("sts").get_caller_identity()["Account"]
        s3_client = self.boto_session.client("s3")
        region = self.boto_session.region_name

        # write to s3 bucket
        manifest_bucket_name = "sagemaker-{}-{}".format(region, account)
        timstamp = str(int(time.time()))
        manifest_s3_file_key = f"{self.experiment_id}/manifest_files/manifest-{timstamp}"
        body = b''
        body += str(json.dumps(manifest_file, sort_keys=True, indent=4)).encode('utf_8')
        try:
            s3_client.put_object(Body=body,
                                 Bucket=manifest_bucket_name,
                                 Key=manifest_s3_file_key)

        except ClientError as e:
            error_code = e.response['Error']['Code']
            message = e.response['Error']['Message']
            raise RuntimeError("Failed to upload manifest data with error {}: {}".format(
                error_code, message
            ))
            
        manifest_file_path = f"s3://{manifest_bucket_name}/{manifest_s3_file_key}"
        logger.info(f"Successfully upload manifest file to s3 bucket path `{manifest_file_path}'")
        return manifest_file_path

    def _generate_manifest(self, input_data_path_list):
        """Generate manifest file and upload it to S3 bucket
        
        Args:
            input_data_path_list (list): A list of strings representing
                input S3 data paths
        
        Returns:
            str: S3 data path for the uploaded manifest file
        """
        # Given a list of S3 buckets, generate a manifest file
        # [
        # {"prefix": "s3://customer_bucket/some/prefix/"},
        # "relative/path/to/data-1",
        # "relative/path/data-2",
        # ...
        # ]
        manifest = []
        shared_prefix, key_path_list = self._get_prefix_and_relative_path(input_data_path_list)
        logger.info(f"Generating manifest file with shared prefix '{shared_prefix}/' ...")
        manifest.append({'prefix': shared_prefix + '/'})
        for relative_key_path in key_path_list:
            manifest.append(relative_key_path)

        manifest_file_path = self._write_manifest_to_s3(manifest_file=manifest)
        return manifest_file_path

    @property
    def last_trained_model_id(self):
        if self.experiment_record._last_trained_model_id is None:
            logger.warning("No model has been trained. Please check later.")

        if self.experiment_record._training_state is not None and \
            self.experiment_record._training_state.endswith("ING"):
            logger.warning(f"A training job with model id '{self.experiment_record._next_model_to_train_id}' "
                           f"is running in state of '{self.experiment_record._training_state}'")

        return self.experiment_record._last_trained_model_id

    @property
    def last_evaluation_job_id(self):
        if self.experiment_record._last_evaluation_job_id is None:
            logger.warning("No model has been evaluated. Please check later.")

        if self.experiment_record._evaluation_state is not None \
            and self.experiment_record._evaluation_state.endswith("ING"):
            logger.warning(f"A evaluation job with job id '{self.experiment_record._next_evaluation_job_id}' "
                           f"is running in state of '{self.experiment_record._evaluation_state}'")

        return self.experiment_record._last_evaluation_job_id

    @property
    def last_hosted_model_id(self):
        if self.experiment_record._last_hosted_model_id is None:
            logger.warning("No model has been hosted. Please deploy a model and check later.")

        if self.experiment_record._hosting_state is not None \
            and self.experiment_record._hosting_state.endswith("ING"):
            logger.warning(f"A deployment with model id '{self.experiment_record._next_model_to_host_id}' "
                           f"is running in state of '{self.experiment_record._hosting_state}'")

        return self.experiment_record._last_hosted_model_id

    @property
    def last_joined_job_id(self):
        if self.experiment_record._last_joined_job_id is None:
            logger.warning("No joining job has been completed. Please check later.")

        if self.experiment_record._joining_state is not None \
            and self.experiment_record._joining_state.endswith("ING"):
            logger.warning(f"A joining job with job id '{self.experiment_record._next_join_job_id}' "
                           f"is running in state of '{self.experiment_record._joining_state}'")

        return self.experiment_record._last_joined_job_id

    @property
    def last_joined_job_train_data(self):        
        record = self.join_db_client.get_join_job_record(self.experiment_id, self.last_joined_job_id)
        return record["output_joined_train_data_s3_path"]

    @property
    def last_joined_job_eval_data(self):
        record = self.join_db_client.get_join_job_record(self.experiment_id, self.last_joined_job_id)
        return record["output_joined_eval_data_s3_path"]

    def _get_hosting_environ_vars(self, model_id):
        """Return hosting endpoint environment variables
        
        Args:
            model_id (str): A unique string representing which model
                to be hosted by the endpoint
        
        Returns:
            dict: A dictionary containing environment variables of hosting endpoint
        """
        environ_vars = {"AWS_DEFAULT_REGION": self._region_name,
                        "EXPERIMENT_ID": self.experiment_id,
                        "EXP_METADATA_DYNAMO_TABLE": self.resource_manager.exp_db_table_name,
                        "MODEL_METADATA_DYNAMO_TABLE": self.resource_manager.model_db_table_name,
                        "MODEL_ID": model_id,
                        "AWS_REGION": self._region_name,
                        "FIREHOSE_STREAM": None,
                        # Set to true if inference logging is required.
                        "LOG_INFERENCE_DATA": str(not self.local_mode).lower(),
                        # For efficient soft model updates.
                        "MODEL_METADATA_POLLING": str(self.soft_deployment).lower()
                        }
        return environ_vars

    def _setup_hosting_endpoint(self, model_id, wait, **kwargs):
        """Initiate a hosting endpoint deployment
        
        Args:
            model_id (str): A unique string representing which model to deploy
            wait (bool): Whether to wait until the deployment finished
        """
        # this should only be called once per experiment
        model_record = self.model_db_client.get_model_record(self.experiment_id, model_id)

        # create resource for firehost stream if not running in 'local' mode
        environ_vars = self._get_hosting_environ_vars(model_id)
        if not self.local_mode:
            stream_name = self.experiment_id
            s3_prefix = f"{self.experiment_id}/inference_data"
            self.resource_manager.create_firehose_stream_if_not_exists(stream_name, s3_prefix)
            environ_vars["FIREHOSE_STREAM"] = stream_name

        sagemaker_model = sagemaker.model.Model(
            image=self.image,
            role=self.resource_manager.iam_role_arn,
            name=model_id,
            model_data=model_record["s3_model_output_path"],
            sagemaker_session=self.sagemaker_session,
            env=environ_vars,
            **kwargs)

        hosting_instance_count = self.resource_manager.hosting_fleet_config.get("instance_count", 1)
        hosting_instance_type = self.resource_manager.hosting_fleet_config.get("instance_type", "local")

        try:
            sagemaker_model.deploy(initial_instance_count=hosting_instance_count,
                                instance_type=hosting_instance_type,
                                endpoint_name=self.experiment_id,
                                wait=wait)
        except Exception as e:
            logger.error(f"Failed to deploy experiment {self.experiment_id}: " + str(e))
            raise UnhandledWorkflowException( "Some error occurred while setting up hosting endpoint. "
                                              "Please check SageMaker console for more information.")

    def _update_model_in_endpoint(self, soft_deploy, model_id, wait=True):
        """Update the model hosted in an existing endpoint
        
        Args:
            soft_deploy (bool): Whether to update the model hosted by the
                endpoint with soft deployment support
            model_id (str): A unique string representing the new model
                to deploy/update
        """
        # update 'next_model_to_host_id' and 'hosting_state'
        self.exp_db_client.update_experiment_next_model_to_host_id(
            self.experiment_id, model_id
        )
        self.exp_db_client.update_experiment_hosting_state(
            self.experiment_id, HostingState.PENDING
        )
        # soft deployment will happen once the 'next_model_host_id' is persisted into ExperimentDB
        if not soft_deploy:
            update_endpoint = True
            environ_vars = self._get_hosting_environ_vars(model_id)
            if not self.local_mode:
                # do SageMaker blue-green deployment
                stream_name = self.experiment_id
                self.resource_manager.create_firehose_stream_if_not_exists(stream_name, self.experiment_id)
                environ_vars["FIREHOSE_STREAM"] = stream_name
            else:
                # close the current container and re-deploy
                update_endpoint = False
                self.sagemaker_session.delete_endpoint_config(self.experiment_id)
                self.sagemaker_session.delete_endpoint(self.experiment_id)
                present, closed = self._close_existing_containers()
                if present:
                    if closed:
                        logger.info("Closed docker container[s] that was already running (maybe from previous job)")
                    else:
                        logger.exception("Failed to close a docker container that was already running (maybe from  "
                                         "previous job). Please close it manually and retry.")

            model_record = self.model_db_client.get_model_record(self.experiment_id, model_id)
            sagemaker_model = sagemaker.model.Model(
                image=self.image,
                role=self.resource_manager.iam_role_arn,
                name=model_id,
                model_data=model_record["s3_model_output_path"],
                sagemaker_session=self.sagemaker_session,
                env=environ_vars)
            hosting_instance_count = self.resource_manager.hosting_fleet_config.get("instance_count", 1)
            hosting_instance_type = self.resource_manager.hosting_fleet_config.get("instance_type", "local")
            try:
                sagemaker_model.deploy(initial_instance_count=hosting_instance_count,
                                    instance_type=hosting_instance_type,
                                    endpoint_name=self.experiment_id,
                                    update_endpoint=update_endpoint,
                                    wait=wait)
            except Exception as e:
                logger.error(e)
                pass

    def _check_if_model_ready(self, model_id):
        """Check if the model exists and already trained
        
        Args:
            model_id (str): A unique string representing which model
                to check
        
        Returns:
            bool: Whether the model exists and is already trained
        """
        # check model_id is not None
        if model_id is None:
            logger.error("Provided model id is None. Please provide valid model id.")
            return False

        # check if the model training is completed successfully to consume by next step
        model_exist = self.model_db_client.check_model_record_exists(
            self.experiment_id, model_id
        )
        if not model_exist:
            logger.error(f"Model with mode_id '{model_id}' was not found in model table. "
                         "Please create a model first")
            return False

        # 'model_id' found in table, check if the 'model_id' is trained
        model_to_deploy = ModelManager(
            model_db_client=self.model_db_client,
            experiment_id=self.experiment_id,
            model_id=model_id
            )

        if not model_to_deploy.model_record.is_train_completed():
            logger.warning(f"Model '{model_id}' is in status of "
            f"{model_to_deploy.model_record._train_state}, Please check later.")
            return False

        return True

    def deploy_model(self, model_id, wait=True, **kwargs):
        """Deploy a new model by creating a new hosting endpoint
        or update the model hosted by an existing endpoint
        
        Args:
            model_id (str): A unique string representing which model
                to deploy/update
            wait (bool): Whether to wait until the deployment finish
        """
        # TODO: add validation/instructions if multiple deployment
        # request happened in th same experiment

        # Sync experiment state if required
        self._sync_experiment_state_with_ddb()

        # check if 'model_id' is already hosted
        if self.experiment_record._last_hosted_model_id == model_id \
            and self.experiment_record._hosting_state == HostingState.DEPLOYED:
            logger.info(f"Model {model_id} is already being hosted. No deployment needed.")
            return

        # No deployment if the given model is not ready
        if not self._check_if_model_ready(model_id):
            return

        # given model is in state of 'Completed', ready to deploy
        logger.info(f"Model '{model_id}' is ready to deploy.")

        # checking hosting workflow state
        if self.experiment_record._hosting_endpoint is None:

            if self.local_mode:
                present, closed = self._close_existing_containers()
                if present:
                    if closed:
                        logger.info("Closed docker container[s] that was already running (maybe from previous job).")
                    else:
                        logger.exception("Failed to close a docker container that was already running (maybe from  "
                                         "previous job). Please close it manually and retry.")
            else:
                logger.info("No hosting endpoint found, creating a new hosting endpoint.")

            # update 'next_model_to_host_id' and 'hosting_state'
            self.exp_db_client.update_experiment_next_model_to_host_id(
                self.experiment_id, model_id
            )
            self.exp_db_client.update_experiment_hosting_state(
                self.experiment_id, HostingState.PENDING
            )

            # starting hosting endpoint
            try:
                self._setup_hosting_endpoint(model_id, wait=wait, **kwargs)
            except Exception as e:
                logger.error(e)
                pass
        else:
            if self.experiment_record._hosting_state.endswith("ING"):
                logger.warning("Some deployment request is in progress, canceled this one")
                return
            elif self.experiment_record._hosting_state.endswith("ED"):
                self._update_model_in_endpoint(self.soft_deployment, model_id, wait=wait)

        # wait until exp ddb table updated
        if self.local_mode or wait:
            deployed_state = self.experiment_record._hosting_state == HostingState.DEPLOYED \
                             and self.experiment_record._last_hosted_model_id == model_id \
                             and self.experiment_record._next_model_to_host_id is None
            num_retries = 0
            num_retries_blue_green_deployment = 0
            
            while not deployed_state:
                # Sync experiment state if required
                # local mode is fast, 'num_retries' increases exponentially
                self._sync_experiment_state_with_ddb()
                logger.debug("Waiting for experiment table hosting status to be updated...")
                
                if self.soft_deployment:
                    time.sleep(2 * (2**num_retries))
                    deployed_state = self.experiment_record._hosting_state == HostingState.DEPLOYED \
                                    and self.experiment_record._last_hosted_model_id == model_id \
                                    and self.experiment_record._next_model_to_host_id is None
                    num_retries += 1
                    if num_retries >=5 and self.local_mode: 
                        raise UnhandledWorkflowException(f"Deployment with model "
                        f"'{self.experiment_record._next_model_to_host_id}' was in "
                        f"state of '{self.experiment_record._hosting_state}'. Failed "
                        "to sync table status.")
                else:
                    # blue-green deployment takes ~8 min, retry every 30 seconds
                    time.sleep(30)
                    deployed_state = self.experiment_record._hosting_state == HostingState.DEPLOYED \
                                    and self.experiment_record._last_hosted_model_id == model_id \
                                    and self.experiment_record._next_model_to_host_id is None
                    num_retries_blue_green_deployment += 1
                    
                    if num_retries_blue_green_deployment%2 == 0:
                        logger.debug(f"Waited {int(num_retries_blue_green_deployment / 2)} " 
                                     f"minutes for blue-green deployment...")

                    if num_retries_blue_green_deployment >=30: # restrict maximum wait time to 15min
                        raise UnhandledWorkflowException(f"Deployment with model "
                        f"'{self.experiment_record._next_model_to_host_id}' was in "
                        f"state of '{self.experiment_record._hosting_state}'. Failed "
                        "to sync table status.")

                if self.experiment_record._hosting_state == HostingState.FAILED:
                    raise SageMakerHostingException("Deployment with model "
                    f"'{self.experiment_record._next_model_to_host_id}' ended "
                    f"with state '{self.experiment_record._hosting_state}'. "
                    "Please check Sagemaker log for more information.")

    @property
    def predictor(self):
        if self.experiment_record._hosting_endpoint:
            return Predictor(endpoint_name=self.experiment_id,
                             sagemaker_session=self.sagemaker_session)
        else:
            logger.warning("Hosting endpoint is not ready yet. A deployment "
                           f"with model id '{self.experiment_record._next_model_to_host_id}' is in state of "
                           f"'{self.experiment_record._hosting_state}'. Please check later.")

            return None

    def ingest_rewards(self, rewards_buffer):
        """Upload rewards data in a rewards buffer to S3 bucket
        
        Args:
            rewards_buffer (list): A list of json blobs containing
                rewards data
        
        Returns:
            str: S3 data prefix path that contains the rewards file
        """
        # use sagemaker-{region}-{account_id} bucket to store reward data
        rewards_bucket_name = self.resource_manager._create_s3_bucket_if_not_exist("sagemaker")
        timstamp = str(int(time.time()))
        rewards_s3_file_key = f"{self.experiment_id}/rewards_data/{self.experiment_id}-{timstamp}/rewards-{timstamp}"
        body = b''

        for reward in rewards_buffer:
            body += str(json.dumps(reward) + '\n').encode('utf_8')

        try:
            self.s3_client.put_object(Body=body,
                                      Bucket=rewards_bucket_name,
                                      Key=rewards_s3_file_key)
        except ClientError as e:
            error_code = e.response['Error']['Code']
            message = e.response['Error']['Message']
            raise RuntimeError("Failed to upload rewards data with error {}: {}".format(
                error_code, message
            ))

        rewards_file_path = f"s3://{rewards_bucket_name}/{rewards_s3_file_key}"

        logger.info("Waiting for reward data to be uploaded.")
        waiter = self.s3_client.get_waiter('object_exists')
        waiter.wait(Bucket=rewards_bucket_name, Key=rewards_s3_file_key)

        logger.info(f"Successfully upload reward files to s3 bucket path {rewards_file_path}")

        reward_s3_prefix = '/'.join(rewards_file_path.split('/')[:-1])

        return reward_s3_prefix

    def ingest_joined_data(self, joined_data_buffer, ratio=0.8):
        """Upload joined data in joined data buffer to S3 bucket
        
        Args:
            joined_data_buffer (list): A list of json blobs containing
                joined data
            ratio (float): Split ratio to split data into
                training data and evaluation data
        """
        # local join to  simulate a joining workflow

        # update next_join_job_id and joining state
        next_join_job_id = JoinManager.name_next_join_job(experiment_id=self.experiment_id)
        self.exp_db_client.update_experiment_next_join_job_id(
            self.experiment_id,
            next_join_job_id)
        self.exp_db_client.update_experiment_joining_state(
            self.experiment_id,
            JoiningState.PENDING)

        self.next_join_job = JoinManager(join_db_client=self.join_db_client,
                                    experiment_id=self.experiment_id,
                                    join_job_id=next_join_job_id,
                                    input_obs_data_s3_path="local-join-does-not-apply",
                                    input_reward_data_s3_path="local-join-does-not-apply",
                                    boto_session=self.boto_session)
    
        logger.info("Started dummy local joining job...")
        self.next_join_job.start_dummy_join(joined_data_buffer=joined_data_buffer,
                                       ratio=ratio)

        # this method can be invoked either in local/SM mode
        succeeded_state = self.experiment_record._joining_state == JoiningState.SUCCEEDED \
                            and self.experiment_record._last_joined_job_id == next_join_job_id \
                            and self.experiment_record._next_join_job_id is None
        num_retries = 0
        
        while not succeeded_state:
            # Sync experiment state if required
            self._sync_experiment_state_with_ddb()
            logger.debug("Waiting for experiment table joining status to be updated...")
            time.sleep(2 * (2**num_retries))
            succeeded_state = self.experiment_record._joining_state == JoiningState.SUCCEEDED \
                                and self.experiment_record._last_joined_job_id == next_join_job_id \
                                and self.experiment_record._next_join_job_id is None
            num_retries += 1
            if num_retries >=5:
                raise UnhandledWorkflowException(f"Joining job '{self.experiment_record._next_join_job_id}' "
                f"was in state of '{self.experiment_record._joining_state}'. Failed to sync table states.")
            if self.experiment_record._joining_state == JoiningState.FAILED or \
                self.experiment_record._joining_state == JoiningState.CANCELLED:
                raise WorkflowJoiningJobException(f"Joining job '{self.experiment_record._next_join_job_id}' "
                f"ended with state '{self.experiment_record._joining_state}'. Please check if provided "
                "joined_data_buffer was in correct data format.")
        
    def join(self, rewards_s3_path, obs_time_window=None, ratio=0.8, wait=True):
        """Start a joining job given rewards data path and observation
        data time window
        
        Args:
            rewards_s3_path (str): S3 data path containing the rewards data
            obs_time_window (int): Define a time window of past X hours to
                select observation data
            ratio (float): Split ratio used to split training data
                and evaluation data
            wait (bool): Whether to wait until the joining job finish
        """
        # Sync experiment state if required
        self._sync_experiment_state_with_ddb()

        if obs_time_window is None:
            logger.warning(f"Start a join job to join reward data "
                           f"under '{rewards_s3_path}' with all the observation data")
            obs_end_time = None
            obs_start_time = None
        else:
            logger.info(f"Start a join job to join reward data "
                        f"under '{rewards_s3_path}' with observation "
                        f"data in the past {obs_time_window} hours")
            obs_end_time = datetime.utcnow()
            obs_start_time = obs_end_time - timedelta(hours=obs_time_window)

        # update next_join_job_id and joining state
        next_join_job_id = JoinManager.name_next_join_job(experiment_id=self.experiment_id)
        self.exp_db_client.update_experiment_next_join_job_id(
            self.experiment_id,
            next_join_job_id)
        self.exp_db_client.update_experiment_joining_state(
            self.experiment_id,
            JoiningState.PENDING)

        input_obs_data_s3_path = f"s3://{self.resource_manager.firehose_bucket}/{self.experiment_id}"
        input_obs_data_s3_path = f"{input_obs_data_s3_path}/inference_data"
        # init joining job, update join table
        logger.info("Creating resource for joining job...")

        try:
            self.next_join_job = JoinManager(join_db_client=self.join_db_client,
                                        experiment_id=self.experiment_id,
                                        join_job_id=next_join_job_id,
                                        input_obs_data_s3_path=input_obs_data_s3_path,
                                        obs_start_time=obs_start_time,
                                        obs_end_time=obs_end_time,
                                        input_reward_data_s3_path=rewards_s3_path,
                                        boto_session=self.boto_session)

            logger.info("Started joining job...")
            self.next_join_job.start_join(ratio=ratio, wait=wait)
        except Exception as e:
            logger.error(e)
            pass

        # wait until exp ddb table updated
        if self.local_mode or wait:
            succeeded_state = self.experiment_record._joining_state == JoiningState.SUCCEEDED \
                              and self.experiment_record._last_joined_job_id == next_join_job_id \
                              and self.experiment_record._next_join_job_id is None
            num_retries = 0
            
            while not succeeded_state:
                # Sync experiment state if required
                self._sync_experiment_state_with_ddb()
                logger.debug("Waiting for experiment table joining status to be updated...")
                time.sleep(2 * (2**num_retries))
                succeeded_state = self.experiment_record._joining_state == JoiningState.SUCCEEDED \
                                  and self.experiment_record._last_joined_job_id == next_join_job_id \
                                  and self.experiment_record._next_join_job_id is None
                num_retries += 1
                if num_retries >=5:
                    raise UnhandledWorkflowException(f"Joining job '{self.experiment_record._next_join_job_id}' "
                    f"was in state of '{self.experiment_record._joining_state}'. Failed to sync table states.")
                if self.experiment_record._joining_state == JoiningState.FAILED or \
                    self.experiment_record._joining_state == JoiningState.CANCELLED:
                    raise WorkflowJoiningJobException(f"Joining job '{self.experiment_record._next_join_job_id}' "
                    f"ended with state '{self.experiment_record._joining_state}'. Please check Athena queries logs "
                    "for more information.")

    def initialize_first_model(self, wait=True, input_data_s3_prefix=None):
        """
        Initializes the first Model training for an Experiment
        
        Args:
            wait (bool): Whether to wait until the training job finishes
            input_data_s3_prefix (str): S3 data path containing data
                used to train the first model
        """
        # Sync experiment state if required
        self._sync_experiment_state_with_ddb()

        # experiment only allow one training job at a time,
        # validate no other training request is in progress
        if self.experiment_record._training_state is not None \
            and self.experiment_record._training_state.endswith("ING"):
            logger.error(f"A training request with model id '{self.experiment_record._next_model_to_train_id}' "
                           f"was in the state of '{self.experiment_record._training_state}'. "
                           "Wait until the training job finished or canceled the request.")
            raise InvalidUsageException("Please wait for old Training Job to Complete before requesting a new one!")
        else:
            # update next_model_to_train_id and training state
            next_model_to_train_id = ModelManager.name_next_model(experiment_id=self.experiment_id)
            logger.info(f"Next Model name would be {next_model_to_train_id}")
            self.exp_db_client.update_experiment_next_model_to_train_id(
                self.experiment_id,
                next_model_to_train_id)
            self.exp_db_client.update_experiment_training_state(
                self.experiment_id,
                TrainingState.PENDING)            
            logger.info(f"Start training job for model '{next_model_to_train_id}''")


            # generate manifest file if input is a list
            manifest_file_path = None
            if isinstance(input_data_s3_prefix, list):
                # generate manifest file and upload to s3
                manifest_file_path = self._generate_manifest(input_data_s3_prefix)

            # init model for training, update model table
            try:
                self.next_model_to_train = ModelManager(
                    model_db_client=self.model_db_client,
                    experiment_id=self.experiment_id,
                    model_id=next_model_to_train_id,
                    image=self.image,
                    role=self.resource_manager.iam_role_arn,
                    instance_config=self.resource_manager.training_fleet_config,
                    boto_session=self.boto_session,
                    algor_config=self.algor_config
                    )

                self.next_model_to_train.fit(
                    wait=wait,
                    input_model_id=None,
                    input_data_s3_prefix=input_data_s3_prefix,
                    manifest_file_path=manifest_file_path,
                    logs=wait
                    )
            except Exception as e:
                logger.error(f"Failed to start new Model Training job for"
                              " ModelId {next_model_to_train_id}")
                logger.error(e)
                pass

        # wait until ExperimentDb state is updated
        if self.local_mode or wait:
            trained_state = self.experiment_record._training_state == TrainingState.TRAINED \
                            and self.experiment_record._last_trained_model_id == next_model_to_train_id \
                            and self.experiment_record._next_model_to_train_id is None
            num_retries = 0
            
            while not trained_state:
                # Sync experiment state if required
                self._sync_experiment_state_with_ddb()
                logger.debug("Waiting for experiment table training status to be updated...")
                time.sleep(2 * (2**num_retries))
                trained_state = self.experiment_record._training_state == TrainingState.TRAINED \
                                and self.experiment_record._last_trained_model_id == next_model_to_train_id \
                                and self.experiment_record._next_model_to_train_id is None
                num_retries += 1
                if num_retries >=5:
                    raise UnhandledWorkflowException(f"Training job '{self.experiment_record._next_model_to_train_id}' "
                    f"was in state of '{self.experiment_record._training_state}'. Expected it to be TRAINED.")
                if self.experiment_record._training_state == TrainingState.FAILED \
                    or self.experiment_record._training_state == TrainingState.STOPPED:
                    raise SageMakerTrainingJobException(f"Training job '{self.experiment_record._next_model_to_train_id}' "
                    f"ended in state of '{self.experiment_record._training_state}'. Please check Sagemaker logs for "
                    "more information.")

    def train_next_model(self, wait=True, input_data_s3_prefix=None, input_model_id=None):
        """
        Train a new model given the training data and a pretrained model
        
        Args:
            wait (bool): Whether to wait until the training finish
            input_data_s3_prefix (str): S3 data path containing data
                used for the training job
            input_model_id (str): A model id to specify which model to
                use as a pre-trained model for the training job
        """
        # Sync experiment state if required
        self._sync_experiment_state_with_ddb()

        # use 'last_trained_model_id' by default as input model for next training
        if input_model_id is None and self.experiment_record._last_trained_model_id is not None:
            logger.info(f"Use last trained model {self.experiment_record._last_trained_model_id} "
                        "as pre-trained model for training")

            input_model_id = self.experiment_record._last_trained_model_id

        if input_model_id != self.experiment_record._last_trained_model_id:
            # No deployment if the given model is not ready
            if not self._check_if_model_ready(input_model_id):
                return

        # experiment only allows one training job at a time,
        # validate no other training request is in progress
        if self.experiment_record._training_state is not None and \
            self.experiment_record._training_state.endswith("ING"):
            logger.error(f"A training request with model id '{self.experiment_record._next_model_to_train_id}' "
                           f"was in the state of '{self.experiment_record._training_state}'. "
                           "Please wait until the training job is finished.")
            raise InvalidUsageException("Please wait for old Training Job to Complete before requesting a new one!")
        else:
            # update next_model_to_train_id and training state
            next_model_to_train_id = ModelManager.name_next_model(experiment_id=self.experiment_id)

            logger.info(f"Starting training job for ModelId '{next_model_to_train_id}''")

            self.exp_db_client.update_experiment_next_model_to_train_id(
                self.experiment_id,
                next_model_to_train_id)
            self.exp_db_client.update_experiment_training_state(
                self.experiment_id,
                TrainingState.PENDING)

            manifest_file_path = None
            if isinstance(input_data_s3_prefix, list):
                # generate manifest file and upload to s3 when having multiple inputs
                manifest_file_path = self._generate_manifest(input_data_s3_prefix)

            try:
                self.next_model_to_train = ModelManager(
                    model_db_client=self.model_db_client,
                    experiment_id=self.experiment_id,
                    model_id=next_model_to_train_id,
                    image=self.image,
                    role=self.resource_manager.iam_role_arn,
                    instance_config=self.resource_manager.training_fleet_config,
                    boto_session=self.boto_session,
                    algor_config=self.algor_config
                    )
                self.next_model_to_train.fit(wait=wait,
                                        input_model_id=input_model_id,
                                        input_data_s3_prefix=input_data_s3_prefix,
                                        manifest_file_path=manifest_file_path,
                                        logs=wait)
            except Exception as e:
                logger.error(e)
                pass

        # wait until exp ddb table updated
        if self.local_mode or wait:
            trained_state = self.experiment_record._training_state == TrainingState.TRAINED \
                            and self.experiment_record._last_trained_model_id == next_model_to_train_id \
                            and self.experiment_record._next_model_to_train_id is None
            num_retries = 0
            
            while not trained_state:
                # Sync experiment state if required
                self._sync_experiment_state_with_ddb()
                logger.debug("Waiting for experiment table training status to be updated...")
                time.sleep(2 * (2**num_retries))
                trained_state = self.experiment_record._training_state == TrainingState.TRAINED \
                                and self.experiment_record._last_trained_model_id == next_model_to_train_id \
                                and self.experiment_record._next_model_to_train_id is None
                num_retries += 1
                if num_retries >=5:
                    raise UnhandledWorkflowException(f"Training job '{self.experiment_record._next_model_to_train_id}' "
                    f"was in state of '{self.experiment_record._training_state}'. Expected it to be TRAINED.")
                if self.experiment_record._training_state == TrainingState.FAILED \
                    or self.experiment_record._training_state == TrainingState.STOPPED:
                    raise SageMakerTrainingJobException(f"Training job '{self.experiment_record._next_model_to_train_id}' "
                    f"ended in state of '{self.experiment_record._training_state}'. Please check Sagemaker logs for "
                    "more information.")

    def evaluate_model(self, input_data_s3_prefix=None, evaluate_model_id=None, wait=True):
        """
        Start an evaluation job to evaluate a model
        
        Args:
            input_data_s3_prefix (str): S3 data path containing data used
                for evaluation
            evaluate_model_id (str): The model used for evaluation
            wait (bool): Whether to wait until the evaluation job finish
        """

        # Sync experiment state if required
        self._sync_experiment_state_with_ddb()

        if evaluate_model_id is None:
            if self.experiment_record._last_trained_model_id:
                # use 'last_trained_model_id' by default as input model for evaluation
                logger.info(f"Using last trained model {self.experiment_record._last_trained_model_id}"
                            "for evaluation")
                evaluate_model_id = self.experiment_record._last_trained_model_id
            else:
                logger.error("Evaluation ModelId in None!")
                pass
        elif evaluate_model_id != self.experiment_record._last_trained_model_id:
            # evaluate_model_id is not None and also not last_trained_model_id
            if not self._check_if_model_ready(evaluate_model_id):
                logger.error(f"ModelId {evaluate_model_id} is not ready for evaluation.")
                evaluate_model_id = None
            else:
                pass
        else:
            # evaluate_model_id is not None and evaluate_model_id == _last_trained_model_id
            pass

        if not evaluate_model_id:
            # evaluate_model_id is still None. Raise an exception...
            raise InvalidUsageException("Please provide a valid ModelId to be evaluated")

        if self.experiment_record._evaluation_state is not None \
            and self.experiment_record._evaluation_state.endswith("ING"):
            logger.warning(f"A evaluation request with job id '{self.experiment_record._next_evaluation_job_id}' "
                f"was in the state of '{self.experiment_record._evaluation_state}'. "
                "Wait until the evaluation job finished or canceled the request.")
            raise InvalidUsageException("Please wait for old Evaluation Job to Complete before requesting a new one!")
        else:
            next_evaluation_job_id = f"{evaluate_model_id}-eval-{str(int(time.time()))}"

            logger.info(f"Evaluating model '{evaluate_model_id}' with evaluation job id '{next_evaluation_job_id}'")

            self.exp_db_client.update_experiment_next_evaluation_job_id(
                self.experiment_id,
                next_evaluation_job_id)

            self.exp_db_client.update_experiment_evaluation_state(
                self.experiment_id,
                EvaluationState.PENDING)

            manifest_file_path = None
            if isinstance(input_data_s3_prefix, list):
                # generate manifest file and upload to s3
                manifest_file_path = self._generate_manifest(input_data_s3_prefix)
            else:
                # add logic if input_data_s3_prefix is string
                pass

            try:
                self.next_model_to_evaluate = ModelManager(
                    model_db_client=self.model_db_client,
                    experiment_id=self.experiment_id,
                    model_id=evaluate_model_id,
                    image=self.image,
                    role=self.resource_manager.iam_role_arn,
                    instance_config=self.resource_manager.evaluation_fleet_config,
                    boto_session=self.boto_session,
                    algor_config=self.algor_config
                    )

                self.next_model_to_evaluate.evaluate(
                    input_data_s3_prefix=input_data_s3_prefix,
                    manifest_file_path=manifest_file_path,
                    evaluation_job_name=next_evaluation_job_id,
                    local_mode = self.local_mode,
                    wait=wait,
                    logs=True
                    )
            except Exception as e:
                logger.error(e)
                pass

        # wait until exp ddb table updated
        if self.local_mode or wait:
            evaluated_state = self.experiment_record._evaluation_state == EvaluationState.EVALUATED \
                              and self.experiment_record._last_evaluation_job_id == next_evaluation_job_id \
                              and self.experiment_record._next_evaluation_job_id is None

            num_retries = 0
            
            while not evaluated_state:
                # Sync experiment state if required
                self._sync_experiment_state_with_ddb()
                logger.debug("Waiting for experiment table evaluation status to be updated...")
                time.sleep(2 * (2**num_retries))
                evaluated_state = self.experiment_record._evaluation_state == EvaluationState.EVALUATED \
                                  and self.experiment_record._last_evaluation_job_id == next_evaluation_job_id \
                                  and self.experiment_record._next_evaluation_job_id is None
                num_retries += 1
                if num_retries >=5:
                    raise UnhandledWorkflowException(f"Evaluation job '{self.experiment_record._next_evaluation_job_id}' "
                    f"was in state of '{self.experiment_record._evaluation_state}'. Failed to sync table states.")
                if self.experiment_record._evaluation_state == EvaluationState.FAILED \
                    or self.experiment_record._evaluation_state == EvaluationState.STOPPED:
                    raise SageMakerTrainingJobException(f"Evaluation job '{self.experiment_record._next_evaluation_job_id}' "
                    f"ended in state of '{self.experiment_record._evaluation_state}'. Please check Sagemaker logs for "
                    "more information.")

    def get_eval_score(self, evaluate_model_id=None, eval_data_path=None):
        """
        Return evaluation score given model id and evaluation data path
        
        Args:
            evaluate_model_id (str): Model id used for evaluation
            eval_data_path (str): S3 data path of evaluation data
        
        Returns:
            float: evaluation score of given model and evaluation data
        """
        # use last trained model by default
        if evaluate_model_id is None:
            evaluate_model_id = self.experiment_record._last_trained_model_id

        if evaluate_model_id != self.experiment_record._last_trained_model_id:
            if not self._check_if_model_ready(evaluate_model_id):
                return
        
        # use last joined job's eval data by default
        if eval_data_path is None:
            eval_data_path = self.last_joined_job_eval_data

        logger.info(f"Getting eval scores for model '{evaluate_model_id}'"
        f" on eval data set '{eval_data_path}'")

        eval_score = "n.a."
        if not evaluate_model_id or not eval_data_path:
            # either evaluate_model_id or eval_data_path is none
            pass
        else:
            model_record = self.model_db_client.get_model_record(self.experiment_id, evaluate_model_id)
            if model_record:
                eval_scores_map = model_record.get('eval_scores', {})
                eval_score = eval_scores_map.get(eval_data_path, eval_score)
            else:
                logger.warn(f"Model Record not found with ModelId: {evaluate_model_id}")
                pass

        if eval_score == "n.a.":
            raise EvalScoreNotAvailableException(f"Evaluation score is not available for model '{evaluate_model_id}'" 
                                                f"with data '{eval_data_path}'.'")
        else:
            eval_score = float(eval_score)
            logger.info(f"Evaluation score for model '{evaluate_model_id}'" 
                f"with data '{eval_data_path}' is {eval_score}.")

        return eval_score
    
    def get_cloudwatch_dashboard_details(self):
        return self.cw_logger.get_cloudwatch_dashboard_details(self.experiment_id)
    
    def clean_resource(self, experiment_id):
        """Clean up resource of the given experiment,
        including hosting endpoint and firehose stream
        """
        if not self.local_mode:
            self.resource_manager.delete_firehose_stream(experiment_id)
            
            # clean athena tables
            logger.info(f"Deleting athena tables for '{experiment_id}'...")
            last_join_job = JoinManager(
                join_db_client=self.join_db_client,
                experiment_id=self.experiment_id,
                join_job_id=self.last_joined_job_id)
            last_join_job._delete_obs_table_if_exist()
            last_join_job._delete_rewards_table_if_exist()
        
        logger.info(f"Deleting hosting endpoint '{experiment_id}'...")
        self.sagemaker_session.delete_endpoint_config(experiment_id)
        self.sagemaker_session.delete_endpoint(experiment_id)

    def clean_table_records(self, experiment_id):
        """Clean up all related records of a given experiment

        Args:
            experiment_id: A unique id reprenting the experiment
                to be cleaned up
        """
        # delete join job records from table
        join_job_records = self.join_db_client.get_all_join_job_records_of_experiment(
            experiment_id
        )

        if join_job_records:
            self.join_db_client.batch_delete_items(
                experiment_id,
                [record["join_job_id"] for record in join_job_records]
            )

        # delete model records from table
        model_records = self.model_db_client.get_all_model_records_of_experiment(
            experiment_id
        )

        if model_records:
            self.model_db_client.batch_delete_items(
                experiment_id,
                [record["model_id"] for record in model_records]
            )

        # # exit sync thread
        self.sync_thread.thread_running.clear()

        # delete exp record from table
        self.exp_db_client.delete_item(experiment_id)

        self._close_existing_containers()

    def _close_existing_containers(self):
        """closing local running containers if exist
        
        Returns:
            (bool, bool): Whether a running container exist,
                Whether successfully close the container
        """
        present = False
        killed = None
        client = docker.from_env()
        running_containers = [i for i in client.containers.list() if self.image in i.image.tags]
        if len(running_containers) > 0:
            present = True
        try:
            for container in running_containers:
                container.kill()
            killed = True
        except Exception as e:
            killed = False
        return present, killed
