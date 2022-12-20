import boto3
import time

region = boto3.Session().region_name
aas_client = boto3.client("application-autoscaling", region_name=region)
sm_client = boto3.client("sagemaker", region_name=region)

def register_scaling(endpoint_name, variant_name, max_capacity, min_capacity=1):
    resource_id = "endpoint/{}/variant/{}".format(endpoint_name, variant_name)
    response = aas_client.register_scalable_target(
        ServiceNamespace='sagemaker',
        ResourceId=resource_id,
        ScalableDimension='sagemaker:variant:DesiredInstanceCount',
        MinCapacity=min_capacity,
        MaxCapacity=max_capacity
    )
    return response


def deregister_scaling(endpoint_name, variant_name):
    resource_id = "endpoint/{}/variant/{}".format(endpoint_name, variant_name)
    response = aas_client.deregister_scalable_target(
        ServiceNamespace='sagemaker',
        ResourceId=resource_id,
        ScalableDimension='sagemaker:variant:DesiredInstanceCount',
    )
    return response


def set_target_scaling_on_invocation(endpoint_name, variant_name, target_value,
                                     scale_out_cool_down=10,
                                     scale_in_cool_down=100):
    policy_name = 'target-tracking-invocations-{}'.format(str(round(time.time())))
    resource_id = "endpoint/{}/variant/{}".format(endpoint_name, variant_name)
    response = aas_client.put_scaling_policy(
        PolicyName=policy_name,
        ServiceNamespace='sagemaker',
        ResourceId=resource_id,
        ScalableDimension='sagemaker:variant:DesiredInstanceCount',
        PolicyType='TargetTrackingScaling',
        TargetTrackingScalingPolicyConfiguration={
            'TargetValue': target_value,
            'PredefinedMetricSpecification': {
                'PredefinedMetricType': 'SageMakerVariantInvocationsPerInstance',
            },
            'ScaleOutCooldown': scale_out_cool_down,
            'ScaleInCooldown': scale_in_cool_down,
            'DisableScaleIn': False
        }
    )
    return policy_name, response


def set_target_scaling_on_cpu_utilization(endpoint_name, variant_name, target_value,
                                          scale_out_cool_down=10,
                                          scale_in_cool_down=100):
    policy_name = 'target-tracking-cpu-util-{}'.format(str(round(time.time())))
    resource_id = "endpoint/{}/variant/{}".format(endpoint_name, variant_name)
    response = aas_client.put_scaling_policy(
        PolicyName=policy_name,
        ServiceNamespace='sagemaker',
        ResourceId=resource_id,
        ScalableDimension='sagemaker:variant:DesiredInstanceCount',
        PolicyType='TargetTrackingScaling',
        TargetTrackingScalingPolicyConfiguration={
            'TargetValue': target_value,
            'CustomizedMetricSpecification':
            {
                'MetricName': 'CPUUtilization',
                'Namespace': '/aws/sagemaker/Endpoints',
                'Dimensions': [
                    {'Name': 'EndpointName', 'Value': endpoint_name},
                    {'Name': 'VariantName', 'Value': variant_name}
                ],
                'Statistic': 'Average',
                'Unit': 'Percent'
            },
            'ScaleOutCooldown': scale_out_cool_down,
            'ScaleInCooldown': scale_in_cool_down,
            'DisableScaleIn': False
        }
    )
    return policy_name, response


def delete_scaling_policies(endpoint_name, variant_name, policy_names=[]):
    resource_id = "endpoint/{}/variant/{}".format(endpoint_name, variant_name)
    for each_policy in policy_names:
        aas_client.delete_scaling_policy(
            PolicyName=each_policy,
            ServiceNamespace='sagemaker',
            ResourceId=resource_id,
            ScalableDimension='sagemaker:variant:DesiredInstanceCount',
        )
        
        
def clear_auto_scaling_and_reset_to_initialCount(endpoint_name, variant_name, intial_count):
    deregister_response = deregister_scaling(endpoint_name=endpoint_name, variant_name=variant_name)
    update_response = sm_client.update_endpoint_weights_and_capacities(EndpointName=endpoint_name,
                                                                       DesiredWeightsAndCapacities=[
                                                                           {
                                                                                'VariantName': variant_name,
                                                                                'DesiredInstanceCount': intial_count
                                                                            },
                                                                        ])

    finished = False
    while not finished:
        endpoint_response = sm_client.describe_endpoint(EndpointName=endpoint_name)
        if endpoint_response["EndpointStatus"] in ["InService", "Deleting", "Failed", "OutOfService"]:
            print("Endpoint {} is in {} state".format(endpoint_name, endpoint_response["EndpointStatus"]))
            finished = True
        else:
            print("Endpoint {} is in updating/creating".format(endpoint_name))
            time.sleep(60)
            
