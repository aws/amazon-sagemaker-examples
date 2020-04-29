import boto3
import logging
import os
import io
import time
import re
import json
from datetime import datetime, timedelta
from threading import Thread
from botocore.exceptions import ClientError
from orchestrator.clients.ddb.join_db_client import JoinDbClient
from orchestrator.workflow.datatypes.join_job_record import JoinJobRecord
from orchestrator.exceptions.ddb_client_exceptions import RecordAlreadyExistsException
from orchestrator.exceptions.workflow_exceptions import UnhandledWorkflowException, \
    JoinQueryIdsNotAvailableException

logger = logging.getLogger("orchestrator")


class JoinManager:
    """A joining job entity with the given experiment. This class
    will handle the joining job creation and joining job metadata
    management.
    """
    def __init__(
            self,
            join_db_client: JoinDbClient,
            experiment_id,
            join_job_id,
            current_state=None,
            input_obs_data_s3_path=None,
            obs_start_time=None,
            obs_end_time=None,
            input_reward_data_s3_path=None,
            output_joined_train_data_s3_path=None,
            output_joined_eval_data_s3_path=None,
            join_query_ids=[],
            boto_session=None):
        """Initialize a joining job entity in the current experiment

        Args:
            join_db_client (JoinDbClient): A DynamoDB client
                to query the joining job table. The 'JoinJob' entity use this client
                to read/update the job state.
            experiment_id (str): A unique id for the experiment. The created/loaded
                joining job will be associated with the experiment.
            join_job_id (str): Aa unique id for the join job. The join job table uses
                join_job_id to manage associated job metadata.
            current_state (str): Current state of the joining job
            input_obs_data_s3_path (str): Input S3 data path for observation data
            obs_start_time (datetime): Datetime object to specify starting time of the
                observation data
            obs_end_time (datetime): Datetime object to specify ending time of the
                observation data
            input_reward_data_s3_path (str): S3 data path for rewards data
            output_joined_train_data_s3_path (str): Output S3 data path for training data split
            output_joined_eval_data_s3_path (str): Output S3 data path for evaluation data split
            join_query_ids (str): Athena join query ids for the joining requests
            boto_session (boto3.session.Session): A session stores configuration
                state and allows you to create service clients and resources.

        Return:
            orchestrator.join_manager.JoinManager: A ``JoinJob`` object associated
            with the given experiment.
        """

        self.join_db_client = join_db_client
        self.experiment_id = experiment_id
        self.join_job_id = join_job_id

        if boto_session is None:
            boto_session = boto3.Session()
        self.boto_session = boto_session

        # formatted athena table name
        self.obs_table_partitioned = self._formatted_table_name(f"obs-{experiment_id}-partitioned")
        self.obs_table_non_partitioned = self._formatted_table_name(f"obs-{experiment_id}")
        self.rewards_table = self._formatted_table_name(f"rewards-{experiment_id}")

        self.query_s3_output_bucket = self._create_athena_s3_bucket_if_not_exist()
        self.athena_client = self.boto_session.client("athena")

        # create a local JoinJobRecord object. 
        self.join_job_record = JoinJobRecord(
            experiment_id,
            join_job_id,
            current_state,
            input_obs_data_s3_path,
            obs_start_time,
            obs_end_time,
            input_reward_data_s3_path,
            output_joined_train_data_s3_path,
            output_joined_eval_data_s3_path,
            join_query_ids
            )

        # create obs partitioned/non-partitioned table if not exists
        if input_obs_data_s3_path and input_obs_data_s3_path != "local-join-does-not-apply":
            self._create_obs_table_if_not_exist()
        # create reward table if not exists
        if input_reward_data_s3_path and input_reward_data_s3_path != "local-join-does-not-apply":
            self._create_rewards_table_if_not_exist()
        # add partitions if input_obs_time_window is not None
        if obs_start_time and obs_end_time:
            self._add_time_partitions(obs_start_time, obs_end_time)

        # try to save this record file. if it throws RecordAlreadyExistsException 
        # reload the record from JoinJobDb, and recreate
        try:
            self.join_db_client.create_new_join_job_record(
                self.join_job_record.to_ddb_record()
            )
        except RecordAlreadyExistsException:
            logger.debug("Join job already exists. Reloading from join job record.")
            join_job_record = self.join_db_client.get_join_job_record(
                experiment_id,
                join_job_id
            )
            self.join_job_record = JoinJobRecord.load_from_ddb_record(join_job_record)
        except Exception as e:
            logger.error("Unhandled Exception! " + str(e))
            raise UnhandledWorkflowException("Something went wrong while creating a new join job")

    def _jsonify(self):
        """Return a jsonify dict with metadata of the 'JoinJob' object
        """
        return self.join_job_record.to_ddb_record()
        
    @classmethod
    def name_next_join_job(cls, experiment_id):
        """Generate unique join job id of a new joining job in the experiment

        Args:
            experiment_id (str): A unique id for the experiment. The created/loaded
                model will be associated with the given experiment.

        Returns:
            str: A unique id for a new joining job
        """
        return experiment_id + "-join-job-id-" + str(int(time.time()))

    def _formatted_table_name(self, table_name_string):
        """Return a formatted athena table name
        Args:
            table_name_string (str): given table name

        Returns:
            str: formatted string
        """
        # athena does not allow special characters other than '_'
        # replace all special characters with '_'
        return re.sub('[^A-Za-z0-9]+', '_', table_name_string)

    def _create_athena_s3_bucket_if_not_exist(self):
        """Create s3 bucket for athena data if not exists
           Use sagemaker-{region}-{account_id} bucket to store data 

        Returns:
            str: s3 bucket name for athena
        """
        account = self.boto_session.client("sts").get_caller_identity()["Account"]
        region = self.boto_session.region_name
        # Use sagemaker bucket to store the joined data
        s3_bucket_name = "sagemaker-{}-{}".format(region, account)

        s3 = self.boto_session.resource("s3")
        s3_client = self.boto_session.client("s3")
        try:
            # 'us-east-1' cannot be specified because it is the default region:
            # https://github.com/boto/boto3/issues/125
            if region == "us-east-1":
                s3.create_bucket(Bucket=s3_bucket_name)
            else:
                s3.create_bucket(
                    Bucket=s3_bucket_name, CreateBucketConfiguration={"LocationConstraint": region}
                )
            logger.info("Successfully create S3 bucket '{}' for athena queries".format(s3_bucket_name))
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            message = e.response["Error"]["Message"]

            if error_code == "BucketAlreadyOwnedByYou":
                pass
            elif (
                error_code == "OperationAborted" and "conflicting conditional operation" in message
            ):
                # If this bucket is already being concurrently created, we don't need to create it again.
                pass
            elif error_code == "TooManyBuckets":
                # Succeed if the default bucket exists
                s3.meta.client.head_bucket(Bucket=s3_bucket_name)
            else:
                raise
        
        s3_waiter = s3_client.get_waiter('bucket_exists')
        s3_waiter.wait(Bucket=s3_bucket_name)
        return s3_bucket_name

    def _create_obs_table_if_not_exist(self):
        """Create athena table for observation data if not exists
        """
        # create both partitioned and non-partitioned table for obs data
        # ensure input path ending with '/'
        input_obs_data_s3_path = self.join_job_record.get_input_obs_data_s3_path()
        input_obs_data_s3_path = input_obs_data_s3_path.strip('/')+'/'

        query_string = f"""
            CREATE EXTERNAL TABLE IF NOT EXISTS {self.obs_table_partitioned} (
                    event_id STRING,
                    action INT,
                    observation STRING,
                    model_id STRING,
                    action_prob FLOAT,
                    sample_prob FLOAT
            )
            PARTITIONED BY (dt string) 
            ROW FORMAT SERDE 'org.openx.data.jsonserde.JsonSerDe'
            LOCATION '{input_obs_data_s3_path}'
        """
        s3_output_path = f"s3://{self.query_s3_output_bucket}/{self.experiment_id}/joined_data/obs_tables"
        query_id = self._start_query(query_string, s3_output_path)
        self.wait_query_to_finish(query_id)

        # non-partitioned-table
        query_string = f"""
            CREATE EXTERNAL TABLE IF NOT EXISTS {self.obs_table_non_partitioned} (
                    event_id STRING,
                    action INT,
                    observation STRING,
                    model_id STRING,
                    action_prob FLOAT,
                    sample_prob FLOAT
            )
            ROW FORMAT SERDE 'org.openx.data.jsonserde.JsonSerDe'
            LOCATION '{input_obs_data_s3_path}'
        """
        s3_output_path = f"s3://{self.query_s3_output_bucket}/{self.experiment_id}/joined_data/obs_tables"
        query_id = self._start_query(query_string, s3_output_path)
        self.wait_query_to_finish(query_id)

        logger.debug(f"Successfully create observation table "
            f"'{self.obs_table_non_partitioned}' and '{self.obs_table_partitioned}' for query")

    def _delete_obs_table_if_exist(self):
        query_string = f"""
            DROP TABLE IF EXISTS {self.obs_table_partitioned}
        """
        s3_output_path = f"s3://{self.query_s3_output_bucket}/{self.experiment_id}/joined_data/obs_tables"
        query_id = self._start_query(query_string, s3_output_path)
        self.wait_query_to_finish(query_id)

        query_string = f"""
            DROP TABLE IF EXISTS {self.obs_table_non_partitioned}
        """
        s3_output_path = f"s3://{self.query_s3_output_bucket}/{self.experiment_id}/joined_data/obs_tables"
        query_id = self._start_query(query_string, s3_output_path)
        self.wait_query_to_finish(query_id)

    def _create_rewards_table_if_not_exist(self):
        """Create athena table for rewards data if not exists
        """
        # create table if not exists
        # ensure input path ending with '/'
        input_reward_data_s3_path = self.join_job_record.get_input_reward_data_s3_path()
        input_reward_data_s3_path = input_reward_data_s3_path.strip('/')+'/'

        query_string = f"""
            CREATE EXTERNAL TABLE IF NOT EXISTS {self.rewards_table} (
                    event_id STRING,
                    reward FLOAT                    
            )
            ROW FORMAT SERDE 'org.openx.data.jsonserde.JsonSerDe'
            LOCATION '{input_reward_data_s3_path}'
        """

        s3_output_path = f"s3://{self.query_s3_output_bucket}/{self.experiment_id}/joined_data/rewards_tables"
        query_id = self._start_query(query_string, s3_output_path)
        self.wait_query_to_finish(query_id)

        logger.debug(f"Successfully create rewards table '{self.rewards_table}' for query")

        query_string = f"""
            ALTER TABLE {self.rewards_table} 
            SET LOCATION '{input_reward_data_s3_path}'
        """

        query_id = self._start_query(query_string, s3_output_path)
        self.wait_query_to_finish(query_id)

        logger.debug(f"Successfully update s3 location of rewards table '{self.rewards_table}'")
    
    def _delete_rewards_table_if_exist(self):
        query_string = f"""
            DROP TABLE IF EXISTS {self.rewards_table}
        """
        s3_output_path = f"s3://{self.query_s3_output_bucket}/{self.experiment_id}/joined_data/rewards_tables"
        query_id = self._start_query(query_string, s3_output_path)
        self.wait_query_to_finish(query_id)

    def _add_time_partitions(self, start_time, end_time):
        """Add partitions to Athena table if not exist

        Args:
            start_time (datetime): Datetime object to specify starting time
                of the observation data
            end_time (datetime): Datetime object to specify ending time
                of the observation data
        """
        input_obs_data_s3_path = self.join_job_record.get_input_obs_data_s3_path()

        # Adding partitions for each hour
        partition_string_list  = []
        time_delta = end_time - start_time
        days = time_delta.days
        seconds = time_delta.seconds
        hours = int(days*24 + seconds/3600)
        for i in range(hours + 1):
            dt =  start_time + timedelta(hours=i)
            dt_str = dt.strftime("%Y-%m-%d-%H")
            bucket_dt_str = dt.strftime("%Y/%m/%d/%H")
            partition_string = f"PARTITION (dt = '{dt_str}') LOCATION '{input_obs_data_s3_path}/{bucket_dt_str}/'"
            partition_string_list.append(partition_string)

        query_string = f"ALTER TABLE {self.obs_table_partitioned} ADD IF NOT EXISTS"
        
        for partition_string in partition_string_list:
            query_string = f"""
            {query_string}\n{partition_string}"""

        s3_output_path = f"s3://{self.query_s3_output_bucket}/{self.experiment_id}/joined_data/partitions"
        query_id = self._start_query(query_string, s3_output_path)
        self.wait_query_to_finish(query_id)
        logger.debug(f"Successfully add partitions to table {self.obs_table_partitioned}")

    def _get_join_query_string(self, ratio=0.8, train_data=True, start_time=None, end_time=None):
        """return query string with given time range and ratio

        Args:
            ratio (float): Split ratio to split training and evaluation data set
            train_data (bool): A boolean value to tell whethere the generated query
                string is for training data
            start_time (datetime): Datetime object to specify starting time
                of the observation data
            end_time (datetime): Datetime object to specify ending time
                of the observation data

        Retrun:
            str: query string for joining
        """
        if start_time is not None:
            start_time_str = start_time.strftime("%Y-%m-%d-%H")
        if end_time is not None:
            end_time_str = end_time.strftime("%Y-%m-%d-%H")

        if start_time is None or end_time is None:
            query_string_prefix = f"""
                    WITH joined_table AS
                    (SELECT {self.obs_table_non_partitioned}.event_id AS event_id,
                            {self.obs_table_non_partitioned}.action AS action,
                            {self.obs_table_non_partitioned}.action_prob AS action_prob,
                            {self.obs_table_non_partitioned}.model_id AS model_id,
                            {self.obs_table_non_partitioned}.observation AS observation,
                            {self.obs_table_non_partitioned}.sample_prob AS sample_prob,
                            {self.rewards_table}.reward AS reward
                    FROM {self.obs_table_non_partitioned}
                    JOIN {self.rewards_table}
                    ON {self.rewards_table}.event_id={self.obs_table_non_partitioned}.event_id)"""
        else:
            query_string_prefix = f"""
                    WITH joined_table AS
                    (   WITH obs_table AS
                        (SELECT *
                         FROM {self.obs_table_partitioned}
                         WHERE dt<='{end_time_str}' AND dt>='{start_time_str}'
                        )
                        SELECT obs_table.event_id AS event_id,
                            obs_table.action AS action,
                            obs_table.action_prob AS action_prob,
                            obs_table.model_id AS model_id,
                            obs_table.observation AS observation,
                            obs_table.sample_prob AS sample_prob,
                            {self.rewards_table}.reward AS reward
                        FROM obs_table
                        JOIN {self.rewards_table}
                        ON {self.rewards_table}.event_id=obs_table.event_id
                    )"""

        if train_data:
            query_sample_string = f"SELECT * FROM joined_table WHERE joined_table.sample_prob <= {ratio}"
        else:
            query_sample_string = f"SELECT * FROM joined_table WHERE joined_table.sample_prob > {ratio}"
        
        query_string = f"""
            {query_string_prefix}
            {query_sample_string}"""
        
        return query_string
        
    def _start_query(self, query_string, s3_output_path):
        """Start query with given query string and output path

        Args:
            query_string (str): Query string to be executed in Athena
            s3_output_path (str): S3 data path to store the output of the Athena query

        Return:
            str: A unique id for Athena query
        """
        # logger.debug(query_string)
        try:
            response = self.athena_client.start_query_execution(
                QueryString=query_string,
                ResultConfiguration={
                    'OutputLocation': s3_output_path,
                    }
                )
            query_id = response['QueryExecutionId']
        except ClientError as e:
            error_code = e.response['Error']['Code']
            message = e.response['Error']['Message']
            raise RuntimeError("Failed to submit athena query with error {}: {}".format(
                error_code, message
            ))
        return query_id

    def wait_query_to_finish(self, query_id):
        """Wait until the Athena query finish

        Args:
            query_id (str): query id of Athena query
        """
        status = 'QUEUED'
        while status == 'RUNNING' or status == 'QUEUED':
            try:
                response = self.athena_client.get_query_execution(
                    QueryExecutionId=query_id
                )
                status = response['QueryExecution']['Status']['State']
                logger.debug(f"Waiting query to finish...")
                time.sleep(5)
            except ClientError as e:
                error_code = e.response['Error']['Code']
                message = e.response['Error']['Message']
                raise RuntimeError("Failed to retrieve athena query status with error {}: {}".format(
                    error_code, message
                ))
    
        if status == 'FAILED':
            raise RuntimeError(f"Query failed with reason: {response['QueryExecution']['Status']['StateChangeReason']}")
        elif status == 'CANCELLED':
            logger.warning("Query was cancelled...")
        elif status == 'SUCCEEDED':
            logger.debug("Query finished successfully")    

    def get_query_status(self, query_id):
        """Return query status given query ID

        Args:
            query_id (str): Query id of Athena query

        Return:
            str: Status of the query
        """
        try:
            response = self.athena_client.get_query_execution(
                    QueryExecutionId=query_id
                )
            status = response['QueryExecution']['Status']['State']
        except ClientError as e:
            error_code = e.response['Error']['Code']
            message = e.response['Error']['Message']
            raise RuntimeError("Failed to retrieve athena query status with error {}: {}".format(
                error_code, message
            ))
        return status
        
    def start_join(self, ratio=0.8, wait=True):
        """Start Athena queries for the joining

        Args:
            ratio (float): Split ratio for training and evaluation data set
            wait (bool): Whether the call should wait until the joining completes.

        """
        logger.info(f"Splitting data into train/evaluation set with ratio of {ratio}")

        obs_start_time, obs_end_time = self.join_job_record.get_obs_start_end_time()

        join_query_for_train_data = self._get_join_query_string(ratio=ratio, 
            train_data=True, start_time=obs_start_time, end_time=obs_end_time)
        join_query_for_eval_data = self._get_join_query_string(ratio=ratio, 
            train_data=False, start_time=obs_start_time, end_time=obs_end_time)

        s3_output_path = f"s3://{self.query_s3_output_bucket}/" \
                f"{self.experiment_id}/joined_data/{self.join_job_id}"
        logger.info(f"Joined data will be stored under {s3_output_path}")


        join_query_id_for_train = self._start_query(join_query_for_train_data, f"{s3_output_path}/train")
        join_query_id_for_eval = self._start_query(join_query_for_eval_data, f"{s3_output_path}/eval")

        # updates join table states vid ddb client
        self.join_db_client.update_join_job_current_state(
            self.experiment_id, self.join_job_id, 'PENDING'
        )
        self.join_db_client.update_join_job_output_joined_train_data_s3_path(
            self.experiment_id, self.join_job_id, f"{s3_output_path}/train"
        )
        self.join_db_client.update_join_job_output_joined_eval_data_s3_path(
            self.experiment_id, self.join_job_id, f"{s3_output_path}/eval"
        )
        self.join_db_client.update_join_job_join_query_ids(
            self.experiment_id, self.join_job_id, [join_query_id_for_train, join_query_id_for_eval]
        )

        if wait:
            self.wait_query_to_finish(join_query_id_for_train)
            self.wait_query_to_finish(join_query_id_for_eval)

    def _val_list_to_csv_byte_string(self, val_list):
        """Convert a list of variables into string in csv format

        Args:
            val_list (list): list of variable names or values

        Return:
            str: A string in csv format, concatenated by ','
        """
        val_str_list = list(map(lambda x: f"\"{x}\"", val_list))
        return str(','.join(val_str_list) + '\n').encode('utf_8')

    def _upload_data_buffer_as_joined_data_format(self, data_buffer, s3_bucket, s3_prefix):
        """Upload joined data buffer to s3 bucket

        Args:
            data_buffer (list): A list of json blobs containing joined data points
            s3_bucket (str): S3 bucket to store the joined data
            s3_prefix (str): S3 prefix path to store the joined data

        Return:
            str: S3 data path of the joined data file
        """
        count = 0
        f = io.BytesIO()
        for record in data_buffer:
            if count == 0:
                f.write(self._val_list_to_csv_byte_string(list(record.keys())))
                count += 1
            f.write(self._val_list_to_csv_byte_string(list(record.values())))
        body = f.getvalue()
        timstamp = str(int(time.time()))
        joined_data_s3_file_key = f"{s3_prefix}/local-joined-data-{timstamp}.csv"
        s3_client = self.boto_session.client("s3")

        try:
            logger.info("_upload_data_buffer_as_joined_data_format put s3://{}/{}".format(
                s3_bucket, joined_data_s3_file_key
            ))
            s3_client.put_object(Body=body,
                                 Bucket=s3_bucket,
                                 Key=joined_data_s3_file_key)
        except ClientError as e:
            error_code = e.response['Error']['Code']
            message = e.response['Error']['Message']
            logger.error("Failed to upload local joined data with error {}: {}".format(
                error_code, message
            ))
            return None

        joined_data_file_path = f"s3://{s3_bucket}/{joined_data_s3_file_key}"

        logger.debug("Waiting for local joined data to be uploaded.")
        waiter = s3_client.get_waiter('object_exists')
        waiter.wait(Bucket=s3_bucket, Key=joined_data_s3_file_key)

        logger.debug(f"Successfully upload local joined data files to s3 bucket path {joined_data_file_path}")

        return joined_data_file_path

    def start_dummy_join(self, joined_data_buffer, ratio=0.8):
        """Start a dummy joining job with the given joined data buffer

        Args:
            joined_data_buffer (list): A list of json blobs containing joined data points
            ratio (float): Split ratio for training and evaluation data set

        """
        logger.info(f"Splitting data into train/evaluation set with ratio of {ratio}")

        joined_train_data_buffer = []
        joined_eval_data_buffer = []

        for record in joined_data_buffer:
            if record["sample_prob"] <= ratio:
                joined_train_data_buffer.append(record)
            else:
                joined_eval_data_buffer.append(record)

        s3_output_path = f"s3://{self.query_s3_output_bucket}/" \
                f"{self.experiment_id}/joined_data/{self.join_job_id}"
        logger.info(f"Joined data will be stored under {s3_output_path}")

        # updates join table states vid ddb client
        self.join_db_client.update_join_job_current_state(
            self.experiment_id, self.join_job_id, 'PENDING'
        )
        self.join_db_client.update_join_job_output_joined_train_data_s3_path(
            self.experiment_id, self.join_job_id, f"{s3_output_path}/train"
        )
        self.join_db_client.update_join_job_output_joined_eval_data_s3_path(
            self.experiment_id, self.join_job_id, f"{s3_output_path}/eval"
        )

        # upload joined data
        joined_train_data_path = self._upload_data_buffer_as_joined_data_format(
            joined_train_data_buffer,
            self.query_s3_output_bucket,
            f"{self.experiment_id}/joined_data/{self.join_job_id}/train")

        joined_eval_data_path = self._upload_data_buffer_as_joined_data_format(
            joined_eval_data_buffer,
            self.query_s3_output_bucket,
            f"{self.experiment_id}/joined_data/{self.join_job_id}/eval")      

        # dummy join finished, update joining job state
        if joined_train_data_path and joined_eval_data_path:
            current_state = "SUCCEEDED"
        else:
            current_state = "FAILED"

        self.join_db_client.update_join_job_current_state(
            self.experiment_id, self.join_job_id, current_state
        )
    
    def update_join_job_state(self):
        for num_retries in range(3):
            try:
                join_job_record = self.join_db_client.get_join_job_record(
                    self.experiment_id, self.join_job_id
                )
                self._update_join_table_states(join_job_record)
            except Exception as e:
                if num_retries >= 2:
                    current_state = 'FAILED'
                    self.join_db_client.update_join_job_current_state(
                        self.experiment_id, self.join_job_id, current_state
                    )
                    logger.error(f"Failing join job '{self.join_job_id}'...")
                    return
                else:
                    logger.warn(f"Received exception '{e}' while updating join "
                    "job status. This exception will be ignored, and retried.")
                    time.sleep(5)
                    continue

    def _update_join_table_states(self, join_job_record):
        """Update the joining job states in the joining job table.
        This method will keep polling the Athena query status and then
        update joining job metadata

        Args:
            join_job_record (dict): Current joining job record in the
                joining table
        """
        if join_job_record is None:
            return
        
        current_state = join_job_record.get("current_state", None)
        join_query_ids = join_job_record.get("join_query_ids", [])

        # join job already ended in terminated state
        if current_state is not None and current_state.endswith("ED"):
            return

        if not join_query_ids:
            raise JoinQueryIdsNotAvailableException(f"Query ids for Joining job "
            f"'{self.join_job_id}' cannot be found.")

        query_states = []

        for query_id in join_query_ids:
            query_states.append(self.get_query_status(query_id))

        # only 'SUCCEEDED' if both queries are 'SUCCEEDED'
        if query_states[0] == 'SUCCEEDED' and query_states[1] == 'SUCCEEDED':
            current_state = 'SUCCEEDED'
        elif 'FAILED' in query_states:
            current_state = 'FAILED'
        elif 'CANCELLED' in query_states:
            current_state = 'CANCELLED'
        else:
            current_state = 'RUNNING'

        # update table states via ddb client
        self.join_db_client.update_join_job_current_state(
            self.experiment_id, self.join_job_id, current_state
        )


            
            

 
