import logging
from boto3.dynamodb.conditions import Key
from orchestrator.exceptions.ddb_client_exceptions import RecordAlreadyExistsException

logger=logging.getLogger(__name__)

class ExperimentDbClient(object):
    def __init__(self, table_session):
        self.table_session = table_session

    def get_experiment_record(self, experiment_id):
        response = self.table_session.query(
            ConsistentRead=True,
            KeyConditionExpression=Key('experiment_id').eq(experiment_id)
        )
        for i in response['Items']:
            return i
        return None

    def create_new_experiment_record(self, record):
        try:
            self.table_session.put_item(
                Item=record,
                ConditionExpression='attribute_not_exists(experiment_id)'
            )
        except Exception as e:
            if "ConditionalCheckFailedException" in str(e):
                raise RecordAlreadyExistsException()
            raise e

    def update_experiment_record(self, record):
        self.table_session.put_item(
            Item=record
        )

    def delete_item(self, experiment_id):
        logger.warning("Deleting experiment record...")
        self.table_session.delete_item(
            Key={
                "experiment_id": experiment_id
            }
        )

    ####  Update states for training workflow
    def update_training_workflow_metadata_with_validation(
            self,
            experiment_id,
            training_workflow_metadata,
            expected_current_next_model_to_train_id
            ):
        '''
        Updates ExperimentDb record for experiment_id with new training_workflow_metadata,
        while validating, next_model_to_train_id is as expected in the old record.
        '''
        self.table_session.update_item(
            Key={'experiment_id': experiment_id},
            UpdateExpression=f'SET training_workflow_metadata = :new_val',
            ConditionExpression='training_workflow_metadata.next_model_to_train_id = :exp_model_id',
            ExpressionAttributeValues={
                ':new_val': training_workflow_metadata,
                ':exp_model_id': expected_current_next_model_to_train_id
                }
        )

    def update_experiment_training_state(self, experiment_id, training_state):
        self.table_session.update_item(
            Key={'experiment_id': experiment_id},
            UpdateExpression=f'SET training_workflow_metadata.training_state = :val',
            ExpressionAttributeValues={':val': training_state}
        )

    def update_experiment_last_trained_model_id(self, experiment_id, last_trained_model_id):
        self.table_session.update_item(
            Key={'experiment_id': experiment_id},
            UpdateExpression=f'SET training_workflow_metadata.last_trained_model_id = :val',
            ExpressionAttributeValues={':val': last_trained_model_id}
        )

    def update_experiment_next_model_to_train_id(self, experiment_id, next_model_to_train_id):
        self.table_session.update_item(
            Key={'experiment_id': experiment_id},
            UpdateExpression=f'SET training_workflow_metadata.next_model_to_train_id = :val',
            ExpressionAttributeValues={':val': next_model_to_train_id}
        )

    ####  Update states for hosting workflow

    def update_experiment_hosting_state(self, experiment_id, hosting_state):
        self.table_session.update_item(
            Key={'experiment_id': experiment_id},
            UpdateExpression=f'SET hosting_workflow_metadata.hosting_state = :val',
            ExpressionAttributeValues={':val': hosting_state}
        )

    def update_experiment_last_hosted_model_id(self, experiment_id, last_hosted_model_id):
        self.table_session.update_item(
            Key={'experiment_id': experiment_id},
            UpdateExpression=f'SET hosting_workflow_metadata.last_hosted_model_id = :val',
            ExpressionAttributeValues={':val': last_hosted_model_id}
        )

    def update_experiment_next_model_to_host_id(self, experiment_id, next_model_to_host_id):
        self.table_session.update_item(
            Key={'experiment_id': experiment_id},
            UpdateExpression=f'SET hosting_workflow_metadata.next_model_to_host_id = :val',
            ExpressionAttributeValues={':val': next_model_to_host_id}
        )

    def update_experiment_hosting_endpoint(self, experiment_id, hosting_endpoint):
        self.table_session.update_item(
            Key={'experiment_id': experiment_id},
            UpdateExpression=f'SET hosting_workflow_metadata.hosting_endpoint = :val',
            ExpressionAttributeValues={':val': hosting_endpoint}
        )

    ####  Update states for joining workflow

    def update_experiment_joining_state(self, experiment_id, joining_state):
        self.table_session.update_item(
            Key={'experiment_id': experiment_id},
            UpdateExpression=f'SET joining_workflow_metadata.joining_state = :val',
            ExpressionAttributeValues={':val': joining_state}
        )

    def update_experiment_last_joined_job_id(self, experiment_id, last_joined_job_id):
        self.table_session.update_item(
            Key={'experiment_id': experiment_id},
            UpdateExpression=f'SET joining_workflow_metadata.last_joined_job_id = :val',
            ExpressionAttributeValues={':val': last_joined_job_id}
        )

    def update_experiment_next_join_job_id(self, experiment_id, next_join_job_id):
        self.table_session.update_item(
            Key={'experiment_id': experiment_id},
            UpdateExpression=f'SET joining_workflow_metadata.next_join_job_id = :val',
            ExpressionAttributeValues={':val': next_join_job_id}
        )

    ####  Update states for evaluation workflow

    def update_experiment_evaluation_state(self, experiment_id, evaluation_state):
        self.table_session.update_item(
            Key={'experiment_id': experiment_id},
            UpdateExpression=f'SET evaluation_workflow_metadata.evaluation_state = :val',
            ExpressionAttributeValues={':val': evaluation_state}
        )

    def update_experiment_last_evaluation_job_id(self, experiment_id, last_evaluation_job_id):
        self.table_session.update_item(
            Key={'experiment_id': experiment_id},
            UpdateExpression=f'SET evaluation_workflow_metadata.last_evaluation_job_id = :val',
            ExpressionAttributeValues={':val': last_evaluation_job_id}
        )

    def update_experiment_next_evaluation_job_id(self, experiment_id, next_evaluation_job_id):
        self.table_session.update_item(
            Key={'experiment_id': experiment_id},
            UpdateExpression=f'SET evaluation_workflow_metadata.next_evaluation_job_id = :val',
            ExpressionAttributeValues={':val': next_evaluation_job_id}
        )
