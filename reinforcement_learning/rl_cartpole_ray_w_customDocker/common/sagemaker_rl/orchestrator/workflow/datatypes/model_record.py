class ModelRecord():
    '''
    This class captures all the data that is needed to run a training job
    for Continuosly Training and Updating models on SageMaker
    '''
    def __init__(
            self,
            experiment_id,
            model_id,
            train_state=None,
            evaluation_job_name=None,
            eval_state=None,
            eval_scores={},
            input_model_id=None,
            input_data_s3_prefix=None,
            manifest_file_path=None,
            eval_data_s3_path=None,
            s3_model_output_path=None,
            training_start_time=None,
            training_end_time=None):

        self.experiment_id = experiment_id
        self.model_id = model_id

        # model table attributes
        self._train_state = train_state
        self._evaluation_job_name = evaluation_job_name
        self._eval_state = eval_state
        self._eval_scores = eval_scores
        self._input_model_id = input_model_id
        self._input_data_s3_prefix = input_data_s3_prefix
        self._manifest_file_path = manifest_file_path
        self._eval_data_s3_path = eval_data_s3_path
        self._s3_model_output_path = s3_model_output_path
        self._training_start_time = training_start_time
        self._training_end_time = training_end_time

    def to_ddb_record(self):
        return {
            'experiment_id': self.experiment_id,
            'model_id': self.model_id,
            'train_state': self._train_state,
            'evaluation_job_name': self._evaluation_job_name,
            'eval_state': self._eval_state,
            'eval_scores': self._eval_scores,
            'input_model_id': self._input_model_id,
            'input_data_s3_prefix': self._input_data_s3_prefix,
            'manifest_file_path': self._manifest_file_path,
            'eval_data_s3_path': self._eval_data_s3_path,
            's3_model_output_path': self._s3_model_output_path,
            'training_start_time': self._training_start_time,
            'training_end_time': self._training_end_time
        }

    @classmethod
    def load_from_ddb_record(cls, record):
        return ModelRecord(
            record["experiment_id"],
            record["model_id"],
            record["train_state"],
            record["evaluation_job_name"],
            record["eval_state"],
            record["eval_scores"],
            record["input_model_id"],
            record["input_data_s3_prefix"],
            record["manifest_file_path"],
            record["eval_data_s3_path"],
            record["s3_model_output_path"],
            record["training_start_time"],
            record["training_end_time"]
            )

    def add_new_training_job_info(
            self,
            input_model_id=None,
            input_data_s3_prefix=None,
            manifest_file_path=None
            ):
        self._input_model_id = input_model_id
        self._input_data_s3_prefix = input_data_s3_prefix
        self._manifest_file_path = manifest_file_path

        # we keep model state as pending, before the SM job has been submitted.
        # the syncer function should update this state, based on SM job status.
        self._train_state = "Pending"
        self._eval_state = None
        self._eval_scores = {}  # eval score for a new model would always be empty.

    def add_new_evaluation_job_info(
            self,
            evaluation_job_name=None,
            eval_data_s3_path=None,
            ):
        self._evaluation_job_name = evaluation_job_name
        self._eval_data_s3_path = eval_data_s3_path

        # we keep evaluation state as pending, before the SM job has been submitted.
        # the syncer function should update this state, based on SM job status.
        self._eval_state = "Pending"

    def get_model_artifact_path(self):
        return self._s3_model_output_path

    def model_in_terminal_state(self):
        if self._train_state:
            return self._train_state.endswith("ed")
        return False

    def update_model_job_status(
            self,
            training_start_time=None,
            training_end_time=None,
            train_state=None,
            s3_model_output_path=None
            ):
        self._training_start_time = training_start_time
        self._training_end_time = training_end_time
        self._train_state =  train_state
        self._s3_model_output_path = s3_model_output_path

    def update_model_as_failed(self):
        self._train_state = "Failed"
    
    def eval_in_terminal_state(self):
        if self._eval_state:
            return self._eval_state.endswith("ed")
        return False

    def add_model_eval_scores(self, eval_score):
        if self._eval_scores is None:
            self._eval_scores = {}
        self._eval_scores[self._eval_data_s3_path] = eval_score
    
    def update_eval_job_state(self, eval_state):
        self._eval_state = eval_state
    
    def update_eval_job_as_failed(self):
        self._eval_state = "Failed"

    def is_train_completed(self):
        if self._train_state and \
                self._train_state == "Completed" and \
                self._s3_model_output_path is not None:
            return True
        return False

    def model_state(self):
        return self._train_state
