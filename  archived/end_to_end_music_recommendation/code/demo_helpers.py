import os
import json
import boto3
import time
import pandas as pd
from sagemaker.lineage.context import Context
from sagemaker.lineage.action import Action
from sagemaker.lineage.association import Association
from sagemaker.lineage.artifact import Artifact
from awscli.customizations.s3.utils import split_s3_bucket_key

def get_data(s3_client, public_s3_data, to_bucket, to_prefix, sample_data=1):
    new_paths = []
    for f in public_s3_data:
        bucket_name, key_name = split_s3_bucket_key(f)
        filename = f.split('/')[-1]
        new_path = "s3://{}/{}/{}".format(to_bucket, to_prefix, filename)
        new_paths.append(new_path)
        
        # only download if not already downloaded
        if not os.path.exists('./data/{}'.format(filename)):
            # download s3 data
            print("Downloading file from {}".format(f))
            s3_client.download_file(bucket_name, key_name, './data/{}'.format(filename))
    
        # subsample the data to create a smaller datatset for this demo
        new_df = pd.read_csv('./data/{}'.format(filename))
        new_df = new_df.sample(frac=sample_data)
        new_df.to_csv('./data/{}'.format(filename), index=False)
        
        # upload s3 data to our default s3 bucket for SageMaker Studio
        print("Uploading {} to {}\n".format(filename, new_path))
        s3_client.upload_file('./data/{}'.format(filename), to_bucket, os.path.join(to_prefix,filename))
        
    return new_paths


def get_model(s3_client, model_path, to_bucket, to_prefix):
    # upload model to our default s3 bucket for SageMaker Studio
    filename = model_path.split('/')[-1]
    print("Uploading {} to {}\n".format(model_path, os.path.join(to_bucket,to_prefix,filename)))
    s3_client.upload_file(model_path, to_bucket, os.path.join(to_prefix,filename))
    return "s://{}".format(os.path.join(to_bucket,to_prefix,filename))
        

def update_data_sources(flow_path, tracks_data_source, ratings_data_source):
    with open(flow_path) as flowf:
        flow = json.load(flowf)
        
    for node in flow['nodes']:
        # if the key exists for our s3 endpoint
        try:
            if node['parameters']['dataset_definition']['name'] == 'tracks.csv':
                # reset the s3 data source for tracks data
                old_source = node['parameters']['dataset_definition']['s3ExecutionContext']['s3Uri']
                print("Changed {} to {}".format(old_source, tracks_data_source))
                node['parameters']['dataset_definition']['s3ExecutionContext']['s3Uri'] = tracks_data_source
            elif node['parameters']['dataset_definition']['name'] == 'ratings.csv':
                # reset the s3 data source for ratings data
                old_source = node['parameters']['dataset_definition']['s3ExecutionContext']['s3Uri']
                print("Changed {} to {}".format(old_source, ratings_data_source))
                node['parameters']['dataset_definition']['s3ExecutionContext']['s3Uri'] = ratings_data_source
        except:
            continue
    # write out the updated json flow file
    with open(flow_path, 'w') as outfile:
        json.dump(flow, outfile)
    
    return flow


def delete_project_resources(sagemaker_boto_client, sagemaker_session, endpoint_names=None, pipeline_names=None, mpg_name=None, 
                             feature_groups=None, prefix='music-recommendation', delete_s3_objects=False, bucket_name=None):
    """Delete AWS resources created during demo.

    Keyword arguments:
    sagemaker_boto_client -- boto3 client for SageMaker used for demo (REQUIRED)
    sagemaker_session     -- sagemaker session used for demo (REQUIRED)
    endpoint_names        -- list of resource names of the model inference endpoint (default None)
    pipeline_names        -- list of resource names of the SageMaker Pipeline (default None)
    mpg_name              -- model package group name (default None)
    feature_groups        -- list of feature group names
    prefix                -- s3 prefix or directory for the demo (default 'music-recommendation')
    delete_s3_objects     -- delete all s3 objects in the demo directory (default False)
    bucket_name           -- name of bucket created for demo (default None)
    """

    def delete_associations(arn):
        # delete incoming associations
        incoming_associations = Association.list(destination_arn=arn)
        for summary in incoming_associations:
            assct = Association(
                source_arn=summary.source_arn,
                destination_arn=summary.destination_arn,
                sagemaker_session=sagemaker_session,
            )
            assct.delete()
            time.sleep(2)

        # delete outgoing associations
        outgoing_associations = Association.list(source_arn=arn)
        for summary in outgoing_associations:
            assct = Association(
                source_arn=summary.source_arn,
                destination_arn=summary.destination_arn,
                sagemaker_session=sagemaker_session,
            )
            assct.delete()
            time.sleep(2)


    def delete_lineage_data():
        for summary in Context.list():
            if prefix in summary.context_name:
                print(f"Deleting context {summary.context_name}")
                delete_associations(summary.context_arn)
                ctx = Context(context_name=summary.context_name, sagemaker_session=sagemaker_session)
                ctx.delete()
                time.sleep(2)

        for summary in Action.list():
            if prefix in summary.source.source_uri:
                print(f"Deleting action {summary.action_name}")
                delete_associations(summary.action_arn)
                actn = Action(action_name=summary.action_name, sagemaker_session=sagemaker_session)
                actn.delete()
                time.sleep(1)

        for summary in Artifact.list():
            if prefix in summary.source.source_uri:
                print(f"Deleting artifact {summary.artifact_arn} {summary.artifact_name}")
                delete_associations(summary.artifact_arn)
                artfct = Artifact(artifact_arn=summary.artifact_arn, sagemaker_session=sagemaker_session)
                artfct.delete()
                time.sleep(1)
                
    # Delete model lineage associations and artifacts created in demo
    try:
        delete_lineage_data()
    except Exception as err:
        print(f"Failed to delete lineage data: {err}")
    
    if endpoint_names is not None:
        try:
            for ep in endpoint_names:
                # must delete monitoring job first on endpoint
                for schedule in sagemaker_boto_client.list_monitoring_schedules(EndpointName=ep)['MonitoringScheduleSummaries']:
                    sagemaker_boto_client.delete_monitoring_schedule(MonitoringScheduleName =schedule['MonitoringScheduleName'])
                time.sleep(30)
                sagemaker_boto_client.delete_endpoint(EndpointName=ep)
                print(f"Deleted endpoint: {ep}")
                
                endpoint_configs = sagemaker_boto_client.list_endpoint_configs(NameContains=ep)
                for endp in endpoint_configs['EndpointConfigs']:
                    sagemaker_boto_client.delete_endpoint_config(EndpointConfigName=endp['EndpointConfigName'])
                    print(f"Deleted endpoint config: {endp['EndpointConfigName']}")
        except Exception as e:
            if f'Could not find endpoint' in e.response.get('Error', {}).get('Message'):
                print(f'Could not find endpoint {ep}')
                pass
            else:
                print(f'Could not delete {ep}')
                pass
                
    
    if pipeline_names is not None:
        for pipeline_name in pipeline_names:
            try:
                sagemaker_boto_client.delete_pipeline(PipelineName=pipeline_name)
                print(f"\nDeleted pipeline: {pipeline_name}")
            except Exception as e:
                if e.response.get('Error', {}).get('Code') == 'ResourceNotFound':
                    print(f'Could not find pipeline {pipeline_name}')
                    pass
                else:
                    print(f'Could not delete {pipeline_name}')
                    pass

    if mpg_name is not None:
        model_packages = sagemaker_boto_client.list_model_packages(ModelPackageGroupName=mpg_name)['ModelPackageSummaryList']
        for mp in model_packages:
            try:
                sagemaker_boto_client.delete_model_package(ModelPackageName=mp['ModelPackageArn'])
                print(f"\nDeleted model package: {mp['ModelPackageArn']}")
                time.sleep(1)

                sagemaker_boto_client.delete_model_package_group(ModelPackageGroupName=mpg_name)
                print(f"\nDeleted model package group: {mpg_name}")
            except Exception as e:
                if 'does not exist' in e.response.get('Error', {}).get('Message'):
                    print(f'Could not find model package group, {mpg_name}')
                    pass
                else:
                    print(f'Could not delete {mpg_name}')
                    pass
    
    models = sagemaker_boto_client.list_models(NameContains=prefix, MaxResults=50)['Models']
    print("\n")
    for m in models:
        sagemaker_boto_client.delete_model(ModelName=m['ModelName'])
        print(f"Deleted model: {m['ModelName']}")
        time.sleep(1)
    
    
    print("\n")
    # delete feature stores within SageMaker Studio
    if feature_groups is not None:
        for fg_name in feature_groups:
            try:
                sagemaker_boto_client.delete_feature_group(FeatureGroupName=fg_name)
                print("Deleted feature group: {}".format(fg_name))
                time.sleep(1)
            except Exception as e:
                if e.response.get('Error', {}).get('Code') == 'ResourceNotFound':
                    print(f'Could not find feature group {fg_name}')
                    pass
                else:
                    print(f'Could not delete {fg_name}')
                    pass
        
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

    
