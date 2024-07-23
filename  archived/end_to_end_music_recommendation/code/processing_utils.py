from time import gmtime, strftime
import boto3

client = boto3.client('sagemaker')

mon_schedule_prefix = 'music-recommender-daily-monitor'
#mon_schedule_name = '{}-endpoint-{}'.format(mon_schedule, datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S"))
def get_last_processing_job():
    
    response = client.list_processing_jobs(
        NameContains='baseline-suggestion-job',
        StatusEquals='Completed',
        SortBy='CreationTime',
        SortOrder='Descending',
        MaxResults=20
    )
    pprint.pprint(response['ProcessingJobSummaries'][0])
    return response['ProcessingJobSummaries'][0]['ProcessingJobName']


def get_endpoints():
    response = client.list_endpoints(
        NameContains='baseline-suggestion-job',
        StatusEquals='Completed',
        SortBy='CreationTime',
        SortOrder='Descending',
        MaxResults=20
    )
    pprint.pprint(response['ProcessingJobSummaries'][0])
    return response['ProcessingJobSummaries'][0]['ProcessingJobName']