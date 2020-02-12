import boto3
import logging
import json

logger = logging.getLogger()
logger.setLevel(logging.INFO)
sm_client = boto3.client('sagemaker')

#Retrieve transform job name from event and return transform job status.
def lambda_handler(event, context):

    if ('TrainingJobName' in event):
        job_name = event['TrainingJobName']

    else:
        raise KeyError('TrainingJobName key not found in function input!'+
                      ' The input received was: {}.'.format(json.dumps(event)))

    #Query boto3 API to check training status.
    try:
        response = sm_client.describe_training_job(TrainingJobName=job_name)
        logger.info("Training job:{} has status:{}.".format(job_name,
            response['TrainingJobStatus']))

    except Exception as e:
        response = ('Failed to read training status!'+ 
                    ' The training job may not exist or the job name may be incorrect.'+ 
                    ' Check SageMaker to confirm the job name.')
        print(e)
        print('{} Attempted to read job name: {}.'.format(response, job_name))

    #We can't marshall datetime objects in JSON response. So convert
    #all datetime objects returned to unix time.
    for index, metric in enumerate(response['FinalMetricDataList']):
        metric['Timestamp'] = metric['Timestamp'].timestamp()

    return {
        'statusCode': 200,
        'trainingMetrics': response['FinalMetricDataList']
    }