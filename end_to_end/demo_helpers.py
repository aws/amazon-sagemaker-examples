import boto3
import time

def delete_project_resources(sagemaker_boto_client, endpoint_name=None, pipeline_name=None, mpg_name=None, prefix='fraud-detect-demo', delete_s3_objects=False, bucket_name=None):
    """Delete AWS resources created during demo.

    Keyword arguments:
    sagemaker_boto_client -- boto3 client for SageMaker used for demo (REQUIRED)
    endpoint_name     -- resource name of the model inference endpoint (default None)
    pipeline_name     -- resource name of the SageMaker Pipeline (default None)
    mpg_name          -- model package group name (default None)
    prefix            -- s3 prefix or directory for the demo (default 'fraud-detect-demo')
    delete_s3_objects -- delete all s3 objects in the demo directory (default False)
    bucket_name       -- name of bucket created for demo (default None)
    """
    
    if endpoint_name is not None:
        try:
            sagemaker_boto_client.delete_endpoint(EndpointName=endpoint_name)
            print(f"Deleted endpoint: {endpoint_name}")
        except Exception as e:
            if 'Could not find endpoint' in e.response.get('Error', {}).get('Message'):
                pass
            else:
                raise(e)
    
    if pipeline_name is not None:
        try:
            sagemaker_boto_client.delete_pipeline(PipelineName=pipeline_name)
            print(f"\nDeleted pipeline: {pipeline_name}")
        except Exception as e:
            if e.response.get('Error', {}).get('Code') == 'ResourceNotFound':
                pass
            else:
                raise(e)
    
    if mpg_name is not None:
        model_packages = sagemaker_boto_client.list_model_packages(ModelPackageGroupName=mpg_name)['ModelPackageSummaryList']
        for mp in model_packages:
            sagemaker_boto_client.delete_model_package(ModelPackageName=mp['ModelPackageArn'])
            print(f"\nDeleted model package: {mp['ModelPackageArn']}")
            time.sleep(1)
    
        try:
            sagemaker_boto_client.delete_model_package_group(ModelPackageGroupName=mpg_name)
            print(f"\nDeleted model package group: {mpg_name}")
        except Exception as e:
            if 'does not exist' in e.response.get('Error', {}).get('Message'):
                pass
            else:
                raise(e)
    
    models = sagemaker_boto_client.list_models(NameContains=prefix, MaxResults=50)['Models']
    print("\n")
    for m in models:
        sagemaker_boto_client.delete_model(ModelName=m['ModelName'])
        print(f"Deleted model: {m['ModelName']}")
        time.sleep(1)
    
    
    feature_groups = sagemaker_boto_client.list_feature_groups(NameContains=prefix)['FeatureGroupSummaries']
    print("\n")
    for fg in feature_groups:
        sagemaker_boto_client.delete_feature_group(FeatureGroupName=fg['FeatureGroupName'])
        print(f"Deleted feature group: {fg['FeatureGroupName']}")
        time.sleep(1)
        
    if delete_s3_objects == True and bucket_name is not None:
        s3 = boto3.resource('s3')
        bucket = s3.Bucket(bucket_name)
        bucket.objects.filter(Prefix=f"{prefix}/").delete()
        print(f"\nDeleted contents of {bucket_name}/{prefix}")
        
def wait_for_feature_group_creation_complete(feature_group):
    status = feature_group.describe().get("FeatureGroupStatus")
    while status == "Creating":
        print("Waiting for Feature Group Creation")
        time.sleep(5)
        status = feature_group.describe().get("FeatureGroupStatus")
    if status != "Created":
        raise RuntimeError(f"Failed to create feature group {feature_group.name}")
    print(f"FeatureGroup {feature_group.name} successfully created.")


class ModelMetrics(object):
    """Accepts model metrics parameters for conversion to request dict."""

    def __init__(
        self,
        model_statistics=None,
        model_constraints=None,
        model_data_statistics=None,
        model_data_constraints=None,
        bias=None,
        explainability=None,
    ):
        """Initialize a ``ModelMetrics`` instance and turn parameters into dict.
        # TODO: flesh out docstrings
        Args:
            model_constraints (MetricsSource):
            model_data_constraints (MetricsSource):
            model_data_statistics (MetricsSource):
            bias (MetricsSource):
            explainability (MetricsSource):
        """
        self.model_statistics = model_statistics
        self.model_constraints = model_constraints
        self.model_data_statistics = model_data_statistics
        self.model_data_constraints = model_data_constraints
        self.bias = bias
        self.explainability = explainability

    def _to_request_dict(self):
        """Generates a request dictionary using the parameters provided to the class."""
        model_metrics_request = {}

        model_quality = {}
        if self.model_statistics is not None:
            model_quality["Statistics"] = self.model_statistics._to_request_dict()
        if self.model_constraints is not None:
            model_quality["Constraints"] = self.model_constraints._to_request_dict()
        if model_quality:
            model_metrics_request["ModelQuality"] = model_quality

        model_data_quality = {}
        if self.model_data_statistics is not None:
            model_data_quality["Statistics"] = self.model_data_statistics._to_request_dict()
        if self.model_data_constraints is not None:
            model_data_quality["Constraints"] = self.model_data_constraints._to_request_dict()
        if model_data_quality:
            model_metrics_request["ModelDataQuality"] = model_data_quality

        if self.bias is not None:
            model_metrics_request["Bias"] = {"Report": self.bias._to_request_dict()}
            #model_metrics_request["Bias"] = self.bias._to_request_dict()
        if self.explainability is not None:
            model_metrics_request["Explainability"] = self.explainability._to_request_dict()
        return model_metrics_request

    
