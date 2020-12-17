import logging
import time

from boto3.dynamodb.conditions import Key
from orchestrator.exceptions.ddb_client_exceptions import RecordAlreadyExistsException

logger=logging.getLogger(__name__)

class ModelDbClient:
    """
    TODO: Deprecate and embed this class in ModelRecord. 
    """
    def __init__(self, table_session):
        self.table_session = table_session

    def check_model_record_exists(self, experiment_id, model_id):
        if self.get_model_record(experiment_id, model_id) is None:
            return False
        else:
            return True

    def get_model_record(self, experiment_id, model_id):
        response = self.table_session.query(
            ConsistentRead=True,
            KeyConditionExpression=Key('experiment_id').eq(experiment_id) & Key('model_id').eq(model_id)
        )
        for i in response['Items']:
            return i
        return None

    def get_model_record_with_retry(self, experiment_id, model_id, retry_gap=5):
        model_record = self.get_model_record(experiment_id, model_id)
        if model_record is None:
            logger.warn("Model Record not found. Waiting for 5 seconds, before retrying.")
            time.sleep(retry_gap)
            return self.get_model_record(experiment_id, model_id)
        return model_record

    def create_new_model_record(self, record):
        try:
            self.table_session.put_item(
                Item=record,
                ConditionExpression='attribute_not_exists(model_id)'
            )
        except Exception as e:
            if "ConditionalCheckFailedException" in str(e):
                raise RecordAlreadyExistsException()
            raise e
    
    def update_model_job_state(self, model_record):
        self.update_model_record(model_record)
    
    def update_model_as_pending(self, model_record):
        # TODO: a model can only be put to pending, from pending state.
        self.update_model_record(model_record)
    
    def update_model_as_failed(self, model_record):
        self.update_model_record(model_record)

    def update_model_eval_job_state(self, model_record):
        # TODO: conditional check to verify model is in *ing state while updating... 
        # Not Trained or some final state.
        self.update_model_record(model_record)

    def update_model_eval_as_pending(self, model_record):
        # TODO: a model eval_state can only be put to pending, from pending state 
        # or a final state. (coz of reruns of evaluation)
        self.update_model_record(model_record)

    def update_model_eval_as_failed(self, model_record):
        # TODO: conditional check to verify model is in *ing state while updating... 
        # Not Trained or some final state.
        self.update_model_record(model_record)

    def update_model_record(self, record):
        self.table_session.put_item(
            Item=record
        )

    def get_all_model_records_of_experiment(self, experiment_id):
        response = self.table_session.query(
            ConsistentRead=True,
            KeyConditionExpression=Key('experiment_id').eq(experiment_id)
        )
        if response['Items']:
            return response['Items']
        else:
            return None

    def batch_delete_items(self, experiment_id, model_id_list):
        logger.warning("Deleting model records of experiment...")
        with self.table_session.batch_writer() as batch:
            for model_id in model_id_list:
                logger.debug(f"Deleting model record '{model_id}'...")
                batch.delete_item(
                    Key={
                        'experiment_id': experiment_id,
                        'model_id': model_id
                    }
                )

    def update_model_input_model_id(self, experiment_id, model_id, input_model_id):
        self.table_session.update_item(
            Key={'experiment_id': experiment_id, 'model_id': model_id},
            UpdateExpression=f'SET input_model_id = :val',
            ExpressionAttributeValues={':val': input_model_id}
        )

    def update_model_input_data_s3_prefix(self, experiment_id, model_id, input_data_s3_prefix):
        self.table_session.update_item(
            Key={'experiment_id': experiment_id, 'model_id': model_id},
            UpdateExpression=f'SET input_data_s3_prefix = :val',
            ExpressionAttributeValues={':val': input_data_s3_prefix}
        )
    def update_model_s3_model_output_path(self, experiment_id, model_id, s3_model_output_path):
        self.table_session.update_item(
            Key={'experiment_id': experiment_id, 'model_id': model_id},
            UpdateExpression=f'SET s3_model_output_path = :val',
            ExpressionAttributeValues={':val': s3_model_output_path}
        )

    def update_model_train_state(self, experiment_id, model_id, train_state):
        self.table_session.update_item(
            Key={'experiment_id': experiment_id, 'model_id': model_id},
            UpdateExpression=f'SET train_state = :val',
            ExpressionAttributeValues={':val': train_state}
        )
    
    def update_model_eval_state(self, experiment_id, model_id, eval_state):
        self.table_session.update_item(
            Key={'experiment_id': experiment_id, 'model_id': model_id},
            UpdateExpression=f'SET eval_state = :val',
            ExpressionAttributeValues={':val': eval_state}
        )

    def update_model_eval_scores(self, experiment_id, model_id, eval_scores):
        self.table_session.update_item(
            Key={'experiment_id': experiment_id, 'model_id': model_id},
            UpdateExpression=f'SET eval_scores = :val',
            ExpressionAttributeValues={':val': eval_scores}
        )

    def update_model_eval_scores_and_state(self, experiment_id, model_id, eval_scores, eval_state):
        self.table_session.update_item(
            Key={'experiment_id': experiment_id, 'model_id': model_id},
            UpdateExpression=f'SET eval_scores = :score_val, eval_state = :state_val',
            ExpressionAttributeValues={
                ':score_val': eval_scores,
                ':state_val': eval_state
            }
        )         

    def update_model_training_start_time(self, experiment_id, model_id, training_start_time):
        self.table_session.update_item(
            Key={'experiment_id': experiment_id, 'model_id': model_id},
            UpdateExpression=f'SET training_start_time = :val',
            ExpressionAttributeValues={':val': training_start_time}
        )

    def update_model_training_end_time(self, experiment_id, model_id, training_end_time):
        self.table_session.update_item(
            Key={'experiment_id': experiment_id, 'model_id': model_id},
            UpdateExpression=f'SET training_end_time = :val',
            ExpressionAttributeValues={':val': training_end_time}
        )

    def update_model_training_stats(self, experiment_id, model_id,
        s3_model_output_path, training_start_time, training_end_time, train_state):
        self.table_session.update_item(
            Key={'experiment_id': experiment_id, 'model_id': model_id},
            UpdateExpression=f"SET s3_model_output_path = :path_val, training_start_time = :start_time_val, "
            f"training_end_time = :end_time_val, train_state = :state_val",
            ExpressionAttributeValues={
                ':path_val': s3_model_output_path,
                ':start_time_val': training_start_time,
                ':end_time_val': training_end_time,
                ':state_val': train_state
            }
        ) 