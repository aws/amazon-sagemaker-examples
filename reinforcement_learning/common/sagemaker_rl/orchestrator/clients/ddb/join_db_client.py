import logging
from boto3.dynamodb.conditions import Key
from orchestrator.exceptions.ddb_client_exceptions import RecordAlreadyExistsException

logger=logging.getLogger(__name__)

class JoinDbClient(object):
    def __init__(self, table_session):
        self.table_session = table_session

    def check_join_job_record_exists(self, experiment_id, join_job_id):
        if self.get_join_job_record(experiment_id, join_job_id) is None:
            return False
        else:
            return True

    def get_join_job_record(self, experiment_id, join_job_id):
        response = self.table_session.query(
            ConsistentRead=True,
            KeyConditionExpression=Key('experiment_id').eq(experiment_id) & Key('join_job_id').eq(join_job_id)
        )
        for i in response['Items']:
            return i
        return None

    def create_new_join_job_record(self, record):
        try:
            self.table_session.put_item(
                Item=record,
                ConditionExpression='attribute_not_exists(join_job_id)'
            )
        except Exception as e:
            if "ConditionalCheckFailedException" in str(e):
                raise RecordAlreadyExistsException()
            raise e

    def update_join_job_record(self, record):
        self.table_session.put_item(
            Item=record
        )

    def get_all_join_job_records_of_experiment(self, experiment_id):
        response = self.table_session.query(
            ConsistentRead=True,
            KeyConditionExpression=Key('experiment_id').eq(experiment_id)
        )
        if response['Items']:
            return response['Items']
        else:
            return None

    def batch_delete_items(self, experiment_id, join_job_id_list):
        logger.warning("Deleting join job records of experiment...")
        with self.table_session.batch_writer() as batch:
            for join_job_id in join_job_id_list:
                logger.debug(f"Deleting join job record {join_job_id}...")
                batch.delete_item(
                    Key={
                        'experiment_id': experiment_id,
                        'join_job_id': join_job_id
                    }
                )

    def update_join_job_current_state(self, experiment_id, join_job_id, current_state):
        self.table_session.update_item(
            Key={'experiment_id': experiment_id, 'join_job_id': join_job_id},
            UpdateExpression=f'SET current_state = :val',
            ExpressionAttributeValues={':val': current_state}
        )

    def update_join_job_input_obs_data_s3_path(self, experiment_id, 
        join_job_id, input_obs_data_s3_path):
        self.table_session.update_item(
            Key={'experiment_id': experiment_id, 'join_job_id': join_job_id},
            UpdateExpression=f'SET input_obs_data_s3_path = :val',
            ExpressionAttributeValues={':val': input_obs_data_s3_path}
        )
        
    def update_join_job_input_reward_data_s3_path(self, experiment_id, 
        join_job_id, input_reward_data_s3_path):
        self.table_session.update_item(
            Key={'experiment_id': experiment_id, 'join_job_id': join_job_id},
            UpdateExpression=f'SET input_reward_data_s3_path = :val',
            ExpressionAttributeValues={':val': input_reward_data_s3_path}
        )

    def update_join_job_join_query_ids(self, experiment_id, join_job_id, join_query_ids):
        self.table_session.update_item(
            Key={'experiment_id': experiment_id, 'join_job_id': join_job_id},
            UpdateExpression=f'SET join_query_ids = :val',
            ExpressionAttributeValues={':val': join_query_ids}
        )

    def update_join_job_obs_end_time(self, experiment_id, join_job_id, obs_end_time):
        self.table_session.update_item(
            Key={'experiment_id': experiment_id, 'join_job_id': join_job_id},
            UpdateExpression=f'SET obs_end_time = :val',
            ExpressionAttributeValues={':val': obs_end_time}
        )

    def update_join_job_obs_start_time(self, experiment_id, join_job_id, obs_start_time):
        self.table_session.update_item(
            Key={'experiment_id': experiment_id, 'join_job_id': join_job_id},
            UpdateExpression=f'SET obs_start_time = :val',
            ExpressionAttributeValues={':val': obs_start_time}
        )

    def update_join_job_output_joined_eval_data_s3_path(self, experiment_id, 
        join_job_id, output_joined_eval_data_s3_path):
        self.table_session.update_item(
            Key={'experiment_id': experiment_id, 'join_job_id': join_job_id},
            UpdateExpression=f'SET output_joined_eval_data_s3_path = :val',
            ExpressionAttributeValues={':val': output_joined_eval_data_s3_path}
        )

    def update_join_job_output_joined_train_data_s3_path(self, experiment_id, 
        join_job_id, output_joined_train_data_s3_path):
        self.table_session.update_item(
            Key={'experiment_id': experiment_id, 'join_job_id': join_job_id},
            UpdateExpression=f'SET output_joined_train_data_s3_path = :val',
            ExpressionAttributeValues={':val': output_joined_train_data_s3_path}
        )