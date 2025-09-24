from IPython.display import Image
from IPython.display import display
from datetime import timedelta
import math
import json
import boto3
from datetime import datetime
from operator import itemgetter

region = boto3.Session().region_name
sm_client = boto3.client("sagemaker", region_name=region)
# Create CloudWatch client
cloudwatch = boto3.client('cloudwatch', region_name=region)


def analysis_inference_recommender_result(job_name, index=0, upper_threshold=55.0, lower_threshold=45.0):
    """
    This function visualizes the benchmarking and derives the target for scaling bases in
    input thresholds.
    """
    inference_recommender_job = sm_client.describe_inference_recommendations_job(JobName=job_name)
    endpoint_name = inference_recommender_job['InferenceRecommendations'][index][
        'EndpointConfiguration']['EndpointName']
    variant_name = inference_recommender_job['InferenceRecommendations'][index][
        'EndpointConfiguration']['VariantName']
    start_time = (inference_recommender_job['CreationTime'] + timedelta(minutes=5)).strftime('%Y-%m-%dT%H:%M:%S.000Z')
    end_time = (inference_recommender_job['LastModifiedTime'] - timedelta(minutes=6)).strftime('%Y-%m-%dT%H:%M:%S.000Z')
    return analysis_and_visualize(endpoint_name, variant_name, start_time, end_time, upper_threshold, lower_threshold)


def analysis_evaluation_result(endpoint_name, variant_name, job_name):
    """
    Visualize the evaluation job result and get the max invocations.
    """
    inference_recommender_job = sm_client.describe_inference_recommendations_job(JobName=job_name)
    start_time = (inference_recommender_job['CreationTime'] + timedelta(minutes=5)).strftime('%Y-%m-%dT%H:%M:%S.000Z')
    end_time = (inference_recommender_job['LastModifiedTime'] - timedelta(minutes=6)).strftime('%Y-%m-%dT%H:%M:%S.000Z')
    stats_response = cloudwatch.get_metric_statistics(
        Period=30,
        StartTime=start_time,
        EndTime=end_time,
        MetricName='Invocations',
        Namespace='AWS/SageMaker',
        Statistics=['Sum'],
        Dimensions=[{'Name': 'EndpointName', 'Value': endpoint_name},
                    {'Name': 'VariantName', 'Value': variant_name}]
    )

    max_value = 0
    for each in stats_response['Datapoints']:
        if max_value < each['Sum']:
            max_value = each['Sum']
            
    invocations_metrics_widget = {
        "metrics": [
            ["AWS/SageMaker", "InvocationModelErrors",
             "EndpointName", endpoint_name,
             "VariantName", variant_name],
            [".", "Invocation5XXErrors", ".", ".", ".", "."],
            [".", "Invocation4XXErrors", ".", ".", ".", "."],
            [".", "ModelLatency", ".", ".", ".", ".", {"yAxis": "right"}],
            [".", "OverheadLatency", ".", ".", ".", ".", {"yAxis": "right"}],
            [".", "InvocationsPerInstance", ".", ".", ".", "."],
            [".", "Invocations", ".", ".", ".", "."]
        ],
        "view": "timeSeries",
        "stat": "ts99",
        "period": 60,
        "title": "InvocationsVsLatencies",
        "width": 1200,
        "height": 400,
        "start": start_time,
        "end": end_time,
    }
    invocations_response = cloudwatch.get_metric_widget_image(
        MetricWidget=json.dumps(invocations_metrics_widget),
        OutputFormat='png'
    )

    display(Image(invocations_response['MetricWidgetImage']))

    return max_value


def analysis_and_visualize(endpoint_name, variant_name, start_time, end_time, upper_threshold=55.0,
                           lower_threshold=45.0, visualize=True):
    """
    This function visualizes the benchmarking and derives the target for scaling bases in
    input thresholds.
    """
    stats_response = cloudwatch.get_metric_statistics(
        Period=30,
        StartTime=start_time,
        EndTime=end_time,
        MetricName='Invocations',
        Namespace='AWS/SageMaker',
        Statistics=['Sum'],
        Dimensions=[{'Name': 'EndpointName', 'Value': endpoint_name},
                    {'Name': 'VariantName', 'Value': variant_name}]
    )

    max_value = 0
    invocations_list = []
    for each in stats_response['Datapoints']:
        invocations_list.append(each)
        if max_value < each['Sum']:
            max_value = each['Sum']

    invocations_list = sorted(invocations_list, key=itemgetter('Timestamp'))

    print("Maximum Invocation seen in benchmarking = {}".format(max_value))
    upper_limit = math.floor(max_value * upper_threshold / 100)
    lower_limit = math.ceil(max_value * lower_threshold / 100)

    print("Invocation upper limit={} for {}%, lower limit={} for {}%".
          format(upper_limit, upper_threshold, lower_limit, lower_threshold))

    max_diff = math.inf
    min_diff = math.inf
    timestamp_upper = datetime.strptime(end_time, '%Y-%m-%dT%H:%M:%S.000Z')
    timestamp_lower = datetime.strptime(start_time, '%Y-%m-%dT%H:%M:%S.000Z')
    for each in invocations_list:
        if abs(each['Sum'] - lower_limit) < min_diff:
            min_diff = abs(each['Sum'] - lower_limit)
            timestamp_lower = each['Timestamp']
        elif abs(each['Sum'] - upper_limit) < max_diff:
            max_diff = abs(each['Sum'] - upper_limit)
            timestamp_upper = each['Timestamp']
        else:
            break

    invocations_metrics_widget = {
        "metrics": [
            ["AWS/SageMaker", "InvocationModelErrors",
             "EndpointName", endpoint_name,
             "VariantName", variant_name],
            [".", "Invocation5XXErrors", ".", ".", ".", "."],
            [".", "Invocation4XXErrors", ".", ".", ".", "."],
            [".", "ModelLatency", ".", ".", ".", ".", {"yAxis": "right"}],
            [".", "OverheadLatency", ".", ".", ".", ".", {"yAxis": "right"}],
            [".", "InvocationsPerInstance", ".", ".", ".", "."],
            [".", "Invocations", ".", ".", ".", "."]
        ],
        "view": "timeSeries",
        "stat": "ts99",
        "period": 60,
        "title": "InvocationsVsLatencies",
        "width": 1200,
        "height": 400,
        "start": start_time,
        "end": end_time,
        "annotations": {
            "horizontal": [
                [
                    {
                        "value": upper_limit
                    },
                    {
                        "value": lower_limit,
                    }
                ]
            ],
            "vertical": [
                [
                    {
                        "value": timestamp_upper.strftime('%Y-%m-%dT%H:%M:%S.000Z')
                    },
                    {
                        "value": timestamp_lower.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
                    }
                ]
            ]
        }
    }
    invocations_response = cloudwatch.get_metric_widget_image(
        MetricWidget=json.dumps(invocations_metrics_widget),
        OutputFormat='png'
    )

    utilization_metrics_widget = {
        "metrics": [
            ["/aws/sagemaker/Endpoints", "CPUUtilization",
             "EndpointName", endpoint_name,
             "VariantName", variant_name],
            [".", "MemoryUtilization", ".", ".", ".", ".", {"yAxis": "right"}],
            [".", "DiskUtilization", ".", ".", ".", ".", {"yAxis": "left"}]
        ],
        "view": "timeSeries",
        "stat": "ts99",
        "period": 60,
        "title": "UtilizationMetrics",
        "width": 1200,
        "height": 400,
        "start": start_time,
        "end": end_time,
        "annotations": {
            "vertical": [
                [
                    {
                        "value": timestamp_upper.strftime('%Y-%m-%dT%H:%M:%S.000Z')
                    },
                    {
                        "value": timestamp_lower.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
                    }
                ]
            ]
        }
    }

    utilization_response = cloudwatch.get_metric_widget_image(
        MetricWidget=json.dumps(utilization_metrics_widget),
        OutputFormat='png'
    )

    x = Image(invocations_response['MetricWidgetImage'])
    y = Image(data=utilization_response['MetricWidgetImage'])

    if visualize:
        display(x, y)

    return max_value, upper_limit, lower_limit

