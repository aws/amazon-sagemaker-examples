import boto3

endpoint_name = "your-model-endpoint" # Your Endpoint name
client = boto3.client("application-autoscaling")
resource_id = f"endpoint/{endpoint_name}/variant/AllTraffic"

# 1. Register scalable target (required to set MinCapacity=0)
client.register_scalable_target(
    ServiceNamespace="sagemaker",
    ResourceId=resource_id,
    ScalableDimension="sagemaker:variant:DesiredInstanceCount",
    MinCapacity=0,  # Allow scale to zero
    MaxCapacity=2
)

# 2. Configure Target Tracking scaling policy
# Logic: Keep "ApproximateBacklogSizePerInstance" (queue size per instance) around 5
# If queue > 5 -> Scale Out (add instances)
# If queue == 0 -> Scale In (remove instances)
client.put_scaling_policy(
    PolicyName="Async-ScaleToZero-Policy",
    ServiceNamespace="sagemaker",
    ResourceId=resource_id,
    ScalableDimension="sagemaker:variant:DesiredInstanceCount",
    PolicyType="TargetTrackingScaling",
    TargetTrackingScalingPolicyConfiguration={
        "TargetValue": 5.0, # Target: no more than 5 pending requests per instance
        # Custom metric for Async Inference
        "CustomizedMetricSpecification": {
            "MetricName": "ApproximateBacklogSizePerInstance",
            "Namespace": "AWS/SageMaker",
            "Dimensions": [{"Name": "EndpointName", "Value": endpoint_name}],
            "Statistic": "Average",
        },
        "ScaleInCooldown": 600,  # Seconds to wait after idle before scaling in (600s = 10 min)
        "ScaleOutCooldown": 60   # Seconds to react when demand increases (keep short)
    }
)

print(f"Done! Endpoint [{endpoint_name}] will now auto scale-in after 10 minutes of idle.")