from datetime import datetime

class JoinJobRecord():
    '''
    This class captures all the data that is needed to run a joining job
    for Continuosly Training and Updating models on SageMaker
    '''
    def __init__(
            self,
            experiment_id,
            join_job_id,
            current_state=None,
            input_obs_data_s3_path=None,
            obs_start_time=None,
            obs_end_time=None,
            input_reward_data_s3_path=None,
            output_joined_train_data_s3_path=None,
            output_joined_eval_data_s3_path=None,
            join_query_ids=[]):

        self.experiment_id = experiment_id
        self.join_job_id = join_job_id

        # join job table attributes
        self._current_state = current_state
        self._input_obs_data_s3_path = input_obs_data_s3_path
        self._obs_start_time = obs_start_time
        self._obs_end_time = obs_end_time
        self._input_reward_data_s3_path = input_reward_data_s3_path
        self._output_joined_train_data_s3_path = output_joined_train_data_s3_path
        self._output_joined_eval_data_s3_path = output_joined_eval_data_s3_path
        self._join_query_ids = join_query_ids

    def to_ddb_record(self):
        obs_start_time_str =  self._obs_start_time.strftime("%Y-%m-%d-%H") if \
            self._obs_start_time is not None else None
        obs_end_time_str = self._obs_end_time.strftime("%Y-%m-%d-%H") if \
            self._obs_end_time is not None else None
        return {
            'experiment_id': self.experiment_id,
            'join_job_id': self.join_job_id,
            'current_state': self._current_state,
            'input_obs_data_s3_path': self._input_obs_data_s3_path,
            'obs_start_time': obs_start_time_str,
            'obs_end_time': obs_end_time_str,
            'input_reward_data_s3_path': self._input_reward_data_s3_path,
            'output_joined_train_data_s3_path': self._output_joined_train_data_s3_path,
            'output_joined_eval_data_s3_path': self._output_joined_eval_data_s3_path,
            'join_query_ids': self._join_query_ids
        }

    @classmethod
    def load_from_ddb_record(cls, record):
        obs_start_time = datetime.strptime(record["obs_start_time"], "%Y-%m-%d-%H") if \
            record["obs_start_time"] is not None else None
        obs_end_time = datetime.strptime(record["obs_end_time"], "%Y-%m-%d-%H") if \
            record["obs_end_time"] is not None else None

        return JoinJobRecord(
            record["experiment_id"],
            record["join_job_id"],
            record["current_state"],
            record["input_obs_data_s3_path"],
            obs_start_time,
            obs_end_time,
            record["input_reward_data_s3_path"],
            record["output_joined_train_data_s3_path"],
            record["output_joined_eval_data_s3_path"],
            record["join_query_ids"]
            )

    def get_input_obs_data_s3_path(self):
        return self._input_obs_data_s3_path

    def get_input_reward_data_s3_path(self):
        return self._input_reward_data_s3_path

    def get_obs_start_end_time(self):
        return self._obs_start_time, self._obs_end_time
