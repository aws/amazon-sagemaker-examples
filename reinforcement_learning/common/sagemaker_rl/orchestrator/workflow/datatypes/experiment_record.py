class ExperimentRecord():
    '''
    This class captures all the data that is needed to run a experiment
    for Continuosly Training and Updating models on SageMaker
    '''
    def __init__(
            self,
            experiment_id,
            training_workflow_metadata={},
            hosting_workflow_metadata={},
            joining_workflow_metadata={},
            evaluation_workflow_metadata={}
            ):
        # unique id common across all experiments in the account
        self.experiment_id = experiment_id

        # training workflow metadata
        self.training_workflow_metadata = training_workflow_metadata
        self._training_state = training_workflow_metadata.get("training_state", None)
        self._last_trained_model_id = training_workflow_metadata.get("last_trained_model_id", None)
        self._next_model_to_train_id = training_workflow_metadata.get("next_model_to_train_id", None)

        # hosting workflow metadata
        self.hosting_workflow_metadata = hosting_workflow_metadata
        self._hosting_state = hosting_workflow_metadata.get("hosting_state", None)
        self._last_hosted_model_id = hosting_workflow_metadata.get("last_hosted_model_id", None)
        self._next_model_to_host_id = hosting_workflow_metadata.get("next_model_to_host_id", None)
        self._hosting_endpoint = hosting_workflow_metadata.get("hosting_endpoint", None)
        
        # joining workflow metadata
        self.joining_workflow_metadata = joining_workflow_metadata
        self._joining_state = joining_workflow_metadata.get("joining_state", None)
        self._last_joined_job_id = joining_workflow_metadata.get("last_joined_job_id", None)
        self._next_join_job_id = joining_workflow_metadata.get("next_join_job_id", None)
        
        # evaluation workflow metadata
        self.evaluation_workflow_metadata = evaluation_workflow_metadata
        self._evaluation_state = evaluation_workflow_metadata.get("evaluation_state", None)
        self._last_evaluation_job_id = evaluation_workflow_metadata.get("last_evaluation_job_id", None)
        self._next_evaluation_job_id = evaluation_workflow_metadata.get("next_evaluation_job_id", None)

    def to_ddb_record(self):
        self.training_workflow_metadata["training_state"] = self._training_state
        self.training_workflow_metadata["last_trained_model_id"] = self._last_trained_model_id
        self.training_workflow_metadata["next_model_to_train_id"] = self._next_model_to_train_id

        self.hosting_workflow_metadata["hosting_state"] = self._hosting_state
        self.hosting_workflow_metadata["last_hosted_model_id"] = self._last_hosted_model_id
        self.hosting_workflow_metadata["next_model_to_host_id"] = self._next_model_to_host_id
        self.hosting_workflow_metadata["hosting_endpoint"] = self._hosting_endpoint

        self.joining_workflow_metadata["joining_state"] = self._joining_state
        self.joining_workflow_metadata["last_joined_job_id"] = self._last_joined_job_id
        self.joining_workflow_metadata["next_join_job_id"] = self._next_join_job_id

        self.evaluation_workflow_metadata["evaluation_state"] = self._evaluation_state
        self.evaluation_workflow_metadata["last_evaluation_job_id"] = self._last_evaluation_job_id
        self.evaluation_workflow_metadata["next_evaluation_job_id"] = self._next_evaluation_job_id

        return {
            'experiment_id': self.experiment_id,
            'training_workflow_metadata': self.training_workflow_metadata,
            'hosting_workflow_metadata': self.hosting_workflow_metadata,
            'joining_workflow_metadata': self.joining_workflow_metadata,
            'evaluation_workflow_metadata': self.evaluation_workflow_metadata
        }

    @classmethod
    def load_from_ddb_record(cls, record):
        return ExperimentRecord(
            record["experiment_id"],
            record["training_workflow_metadata"],
            record["hosting_workflow_metadata"],
            record["joining_workflow_metadata"],
            record["evaluation_workflow_metadata"]
        )