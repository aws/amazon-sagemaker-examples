import pandas as pd
from decimal import Decimal
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

endpoint_metrics = {
    "Invocation4XXErrors": {"Unit": "None", "Statistics": "Sum", "ExtendedStatistics": "None", "Period": 60,
                            "Namespace": "AWS/SageMaker"},
    "Invocation5XXErrors": {"Unit": "None", "Statistics": "Sum", "ExtendedStatistics": "None", "Period": 60,
                            "Namespace": "AWS/SageMaker"},
    "Invocations": {"Unit": "None", "Statistics": "Sum", "ExtendedStatistics": "None", "Period": 60,
                    "Namespace": "AWS/SageMaker"},
    "InvocationsPerInstance": {"Unit": "None", "Statistics": "Sum", "ExtendedStatistics": "None", "Period": 60,
                               "Namespace": "AWS/SageMaker"},
    "ModelLatency": {"Unit": "Microseconds", "Statistics": "None", "ExtendedStatistics": "p99", "Period": 60,
                     "Namespace": "AWS/SageMaker"},
    "OverheadLatency": {"Unit": "Microseconds", "Statistics": "None", "ExtendedStatistics": "p99", "Period": 60,
                        "Namespace": "AWS/SageMaker"},
    "CPUUtilization": {"Unit": "Percent", "Statistics": "Maximum", "ExtendedStatistics": "None", "Period": 60,
                       "Namespace": "/aws/sagemaker/Endpoints"},
    "MemoryUtilization": {"Unit": "Percent", "Statistics": "Maximum", "ExtendedStatistics": "None", "Period": 60,
                          "Namespace": "/aws/sagemaker/Endpoints"},
    "GPUUtilization": {"Unit": "Percent", "Statistics": "Maximum", "ExtendedStatistics": "None", "Period": 60,
                       "Namespace": "/aws/sagemaker/Endpoints"},
    "GPUMemoryUtilization": {"Unit": "Percent", "Statistics": "Maximum", "ExtendedStatistics": "None", "Period": 60,
                             "Namespace": "/aws/sagemaker/Endpoints"},
    "DiskUtilization": {"Unit": "Percent", "Statistics": "Maximum", "ExtendedStatistics": "None", "Period": 60,
                        "Namespace": "/aws/sagemaker/Endpoints"}}

invocation_metrics = ['Invocations', 'ModelLatency', 'OverheadLatency']
invocation_error_metrics = ['Invocation4XXErrors', 'Invocation5XXErrors']
hardware_metrics = ['CPUUtilization', 'MemoryUtilization', 'DiskUtilization']
gpu_metrics = ['GPUUtilization', 'GPUMemoryUtilization']


def get_inference_recommender_job_details(client, job_name):
    job_details = client.describe_inference_recommendations_job(
        JobName=job_name
    )
    return job_details


def get_job_results_as_dataframe(client, job_name):
    job_details = get_inference_recommender_job_details(client, job_name)

    if job_details['Status'] == 'COMPLETED':
        endpoint_names = []
        variant_names = []
        instance_type = []
        initial_count = []
        cost_per_hour = []
        cost_per_inference = []
        max_invocations = []
        model_latency = []
        is_compiled = []
        env_parameters = []

        inference_recommendations = job_details['InferenceRecommendations']
        start_time = job_details['CreationTime']
        end_time = job_details['LastModifiedTime']

        for endpoint_configuration in inference_recommendations:
            endpoint_names.append(endpoint_configuration['EndpointConfiguration']['EndpointName'])
            variant_names.append(endpoint_configuration['EndpointConfiguration']['VariantName'])
            instance_type.append(endpoint_configuration['EndpointConfiguration']['InstanceType'])
            initial_count.append(int(endpoint_configuration['EndpointConfiguration']['InitialInstanceCount']))
            cost_per_hour.append(Decimal(endpoint_configuration['Metrics']['CostPerHour']))
            cost_per_inference.append(Decimal(endpoint_configuration['Metrics']['CostPerInference']))
            max_invocations.append(int(endpoint_configuration['Metrics']['MaxInvocations']))
            model_latency.append(int(endpoint_configuration['Metrics']['ModelLatency']))
#             is_compiled.append(endpoint_configuration['ModelConfiguration']['Compiled'])
            env_parameters.append(endpoint_configuration['ModelConfiguration']
                                  ['EnvironmentParameters'])

        if len(endpoint_names) == 0:
            print(f'No endpoint recommendations for {job_name}')
            exit(-1)

        data = {
            "InstanceType": instance_type,
            "MaximumInvocations": max_invocations,
            "ModelLatency": model_latency,
            "CostPerHour": cost_per_hour,
            "CostPerInference": cost_per_inference,
            "EndpointName": endpoint_names,
            "VariantName": variant_names,
            "InitialCount": initial_count,
            "EnvParameters": env_parameters,
            "StartTime": start_time,
            "EndTime": end_time
        }
        df = pd.DataFrame(data)

        return df
    else:
        print(f"Job {job_name} is in {job_details['Status']} status, we can plot only completed jobs")
        exit(0)


def get_cw_metrics(cw_client, endpoint_name, variant_name, metrics_name, start_time, end_time):
    if endpoint_metrics[metrics_name]['Statistics'] != 'None':
        cw_data = cw_client.get_metric_statistics(
            Namespace=endpoint_metrics[metrics_name]['Namespace'],
            MetricName=metrics_name,
            Dimensions=[
                {
                    'Name': 'EndpointName',
                    'Value': endpoint_name
                },
                {
                    'Name': 'VariantName',
                    'Value': variant_name
                }
            ],
            StartTime=start_time,
            EndTime=end_time,
            Period=endpoint_metrics[metrics_name]['Period'],
            Statistics=[
                endpoint_metrics[metrics_name]['Statistics'],
            ],
            Unit=endpoint_metrics[metrics_name]['Unit']
        )
    elif endpoint_metrics[metrics_name]['ExtendedStatistics'] != 'None':
        cw_data = cw_client.get_metric_statistics(
            Namespace=endpoint_metrics[metrics_name]['Namespace'],
            MetricName=metrics_name,
            Dimensions=[
                {
                    'Name': 'EndpointName',
                    'Value': endpoint_name
                },
                {
                    'Name': 'VariantName',
                    'Value': variant_name
                }
            ],
            StartTime=start_time,
            EndTime=end_time,
            Period=endpoint_metrics[metrics_name]['Period'],
            ExtendedStatistics=[
                endpoint_metrics[metrics_name]['ExtendedStatistics'],
            ],
            Unit=endpoint_metrics[metrics_name]['Unit']
        )
    else:
        print(f'Both ExtendedStatistics & Statistics are None')
        exit(0)

    return cw_data


def get_x_from_datapoints(datapoints):
    timestamps = []

    for datapoint in datapoints:
        timestamps.append(datapoint['Timestamp'])

    return timestamps


def get_y_from_datapoints(datapoints, metric_statistics):
    values = []

    for datapoint in datapoints:
        values.append(datapoint[metric_statistics])

    return values


def get_y_from_extended_datapoints(datapoints, metric_statistics):
    values = []

    for datapoint in datapoints:
        values.append(datapoint['ExtendedStatistics'][metric_statistics])

    return values


def get_unit_from_datapoints(datapoints):
    for datapoint in datapoints:
        return datapoint['Unit']


def sort_cw_datapoints_by_timestamp(datapoints):
    datapoints.sort(key=lambda x: x["Timestamp"])
    return datapoints

def get_endpoint_metrics(sm_client, cw_client, region, job_name, include_plots=False):
    df = get_job_results_as_dataframe(sm_client, job_name)

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'center')
    pd.set_option('display.precision', 3)

    for record in df.to_dict('records'):
        if include_plots:
            fig = plt.figure(figsize=(20, 16), constrained_layout=True)
            fig.suptitle(f"Instance type {record['InstanceType']} Endpoint {record['EndpointName']}",
                         fontsize=16)
            spec = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)
            f_ax1 = fig.add_subplot(spec[0, 0])
            f_ax2 = fig.add_subplot(spec[0, 1])
            f_ax3 = fig.add_subplot(spec[0, 2])
            f_ax4 = fig.add_subplot(spec[1, 0])
            f_ax5 = fig.add_subplot(spec[1, 1])
            f_ax6 = fig.add_subplot(spec[1, 2])
            f_ax7 = fig.add_subplot(spec[2, 0])
            f_ax8 = fig.add_subplot(spec[2, 1])
            f_ax9 = fig.add_subplot(spec[2, 2])

        invocation_data = get_cw_metrics(cw_client, record['EndpointName'], record['VariantName'], 'Invocations',
                                         record['StartTime'], record['EndTime'])
        sorted_invocation_data = sort_cw_datapoints_by_timestamp(invocation_data['Datapoints'])

        if include_plots:
            f_ax1_x = get_x_from_datapoints(sorted_invocation_data)
            f_ax1_y = get_y_from_datapoints(sorted_invocation_data, 'Sum')
            f_ax1.set_title('Invocations')
            f_ax1.set_ylabel('No of Invocations')
            f_ax1.plot(f_ax1_x, f_ax1_y)

        model_latency_data = get_cw_metrics(cw_client, record['EndpointName'], record['VariantName'], 'ModelLatency',
                                            record['StartTime'], record['EndTime'])
        sorted_model_latency_data = sort_cw_datapoints_by_timestamp(model_latency_data['Datapoints'])
        
        if include_plots:
            f_ax2_x = get_x_from_datapoints(sorted_model_latency_data)
            f_ax2_y = get_y_from_extended_datapoints(sorted_model_latency_data, 'p99')
            f_ax2.set_title('ModelLatency')
            f_ax2.set_ylabel(get_unit_from_datapoints(sorted_model_latency_data))
            f_ax2.plot(f_ax2_x, f_ax2_y)

        overhead_latency_data = get_cw_metrics(cw_client, record['EndpointName'], record['VariantName'],
                                               'OverheadLatency', record['StartTime'], record['EndTime'])
        sort_overhead_latency_data = sort_cw_datapoints_by_timestamp(overhead_latency_data['Datapoints'])

        if include_plots:
            f_ax3_x = get_x_from_datapoints(sort_overhead_latency_data)
            f_ax3_y = get_y_from_extended_datapoints(sort_overhead_latency_data, 'p99')
            f_ax3.set_title('OverheadLatency')
            f_ax3.set_ylabel(get_unit_from_datapoints(sort_overhead_latency_data))
            f_ax3.plot(f_ax3_x, f_ax3_y)

        cpu_utilization_data = get_cw_metrics(cw_client, record['EndpointName'], record['VariantName'],
                                              'CPUUtilization', record['StartTime'], record['EndTime'])
        sorted_cpu_utilization_data = sort_cw_datapoints_by_timestamp(cpu_utilization_data['Datapoints'])
        if include_plots:
            f_ax4_x = get_x_from_datapoints(sorted_cpu_utilization_data)
            f_ax4_y = get_y_from_datapoints(sorted_cpu_utilization_data, 'Maximum')
            f_ax4.set_title('CPUUtilization')
            f_ax4.set_ylabel(get_unit_from_datapoints(sorted_cpu_utilization_data))
            f_ax4.plot(f_ax4_x, f_ax4_y)

        memory_utilization_data = get_cw_metrics(cw_client, record['EndpointName'], record['VariantName'],
                                                 'MemoryUtilization', record['StartTime'], record['EndTime'])
        sorted_memory_utilization_data = sort_cw_datapoints_by_timestamp(memory_utilization_data['Datapoints'])
        if include_plots:
            f_ax5_x = get_x_from_datapoints(sorted_memory_utilization_data)
            f_ax5_y = get_y_from_datapoints(sorted_memory_utilization_data, 'Maximum')
            f_ax5.set_title('MemoryUtilization')
            f_ax5.set_ylabel(get_unit_from_datapoints(sorted_memory_utilization_data))
            f_ax5.plot(f_ax5_x, f_ax5_y)

        disk_utilization_data = get_cw_metrics(cw_client, record['EndpointName'], record['VariantName'],
                                               'DiskUtilization', record['StartTime'], record['EndTime'])
        sorted_disk_utilization_data = sort_cw_datapoints_by_timestamp(disk_utilization_data['Datapoints'])
        if include_plots:
            f_ax6_x = get_x_from_datapoints(sorted_disk_utilization_data)
            f_ax6_y = get_y_from_datapoints(sorted_disk_utilization_data, 'Maximum')
            f_ax6.set_title('DiskUtilization')
            f_ax6.set_ylabel(get_unit_from_datapoints(sorted_disk_utilization_data))
            f_ax6.plot(f_ax6_x, f_ax6_y)

        user_error_data = get_cw_metrics(cw_client, record['EndpointName'], record['VariantName'],
                                         'Invocation4XXErrors', record['StartTime'], record['EndTime'])
        sorted_user_error_data = sort_cw_datapoints_by_timestamp(user_error_data['Datapoints'])
        if include_plots:
            f_ax7_x = get_x_from_datapoints(sorted_user_error_data)
            f_ax7_y = get_y_from_datapoints(sorted_user_error_data, 'Sum')
            f_ax7.set_title('Invocation4XXErrors')
            f_ax7.set_ylabel(get_unit_from_datapoints(sorted_user_error_data))
            f_ax7.plot(f_ax7_x, f_ax7_y)

        system_error_data = get_cw_metrics(cw_client, record['EndpointName'], record['VariantName'],
                                           'Invocation5XXErrors', record['StartTime'], record['EndTime'])
        sorted_system_error_data = sort_cw_datapoints_by_timestamp(system_error_data['Datapoints'])
        if include_plots:
            f_ax8_x = get_x_from_datapoints(sorted_system_error_data)
            f_ax8_y = get_y_from_datapoints(sorted_system_error_data, 'Sum')
            f_ax8.set_title('Invocation5XXErrors')
            f_ax8.set_ylabel(get_unit_from_datapoints(sorted_system_error_data))
            f_ax8.plot(f_ax8_x, f_ax8_y)

        per_instance_data = get_cw_metrics(cw_client, record['EndpointName'], record['VariantName'],
                                           'InvocationsPerInstance', record['StartTime'], record['EndTime'])
        sorted_per_instance_data = sort_cw_datapoints_by_timestamp(per_instance_data['Datapoints'])
        if include_plots:
            f_ax9_x = get_x_from_datapoints(sorted_per_instance_data)
            f_ax9_y = get_y_from_datapoints(sorted_per_instance_data, 'Sum')
            f_ax9.set_title('InvocationsPerInstance')
            f_ax9.set_ylabel(get_unit_from_datapoints(sorted_per_instance_data))
            f_ax9.plot(f_ax9_x, f_ax9_y)

    if include_plots:
        plt.show()
        
    return df
