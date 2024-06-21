import argparse
import sys

import boto3

# athena_workgroup = 'SD2ReportsWorkgroup'

# cf_client = boto3.client('cloudformation')
# athena_client = boto3.client('athena')
# sagemaker_client = boto3.client('sagemaker')

# s3_session = boto3.Session()
# s3 = s3_session.resource(service_name='s3')

# Identify the name of the Sagemaker workflow.
# This name will be used to remove it.
def get_a2i_workflow_name(stack_name):
    workflow_name = ""

    try:
        resources = cf_client.list_stack_resources(StackName=stack_name)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

    # Get the nested stack name
    for resource in resources["StackResourceSummaries"]:
        if resource["ResourceType"] == "AWS::CloudFormation::Stack":
            nested_stack_name = resource["PhysicalResourceId"].split(":")[-1].split("/")[1]
            break

    try:
        resources = cf_client.describe_stacks(StackName=nested_stack_name)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

    for parameter in resources["Stacks"][0]["Parameters"]:
        if parameter["ParameterKey"] == "A2iWorkflowName":
            workflow_name = parameter["ParameterValue"]
            break

    return workflow_name


# Identify all of the S3 buckets in the CloudFormation Stack
# The s3_buckets list will have the Physical Resource ID
def get_s3_buckets(stack_name, exclude_buckets=None):
    print(f"Getting S3 buckets for CloudFormation stack {stack_name} and all nested stacks")
    s3_buckets = set()
    if exclude_buckets:
        exclude_buckets = set(exclude_buckets)

    try:
        resources = cf_client.list_stack_resources(StackName=stack_name)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

    for resource in resources["StackResourceSummaries"]:
        if resource["ResourceType"] == "AWS::S3::Bucket":
            if exclude_buckets:
                if resource["PhysicalResourceId"] in exclude_buckets:
                    continue
            print(f"* Identified bucket {resource['PhysicalResourceId']}")
            s3_buckets.add(resource["PhysicalResourceId"])
        if resource["ResourceType"] == "AWS::CloudFormation::Stack":
            nested_stack_name = resource["PhysicalResourceId"].split(":")[-1].split("/")[1]
            nested_buckets = get_s3_buckets(nested_stack_name, exclude_buckets=exclude_buckets)
            s3_buckets.update(nested_buckets)

    return s3_buckets


# Iterate through each of the S3 buckets and remove all objects + versions
def clear_buckets(s3_buckets):
    print(f'Clearing {" ".join(s3_buckets)} S3 buckets')
    for s3_bucket in s3_buckets:
        bucket = s3.Bucket(s3_bucket)
        try:
            print(f"* Removing objects and versions in s3://{s3_bucket}")
            bucket.object_versions.delete()
            result = True
        except Exception as e:
            print(f"* Could not remove objects and versions in s3://{s3_bucket}: {e}")
            result = False
            continue

    return result


# Delete the Sagemaker A2I flow definition
def delete_a2i_flow(flow_name):
    try:
        response = client.delete_flow_definition(FlowDefinitionName="string")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


# Delete the CloudFormation stack
def cleanup_stack(stack_name):
    print(f"Deleting CloudFormation stack {stack_name}")
    try:
        cf_client.delete_stack(StackName=stack_name)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

    waiter = cf_client.get_waiter("stack_delete_complete")

    waiter.wait(StackName=stack_name)

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Salus Cleanup", add_help=False)

    parser.add_argument("--stack-name", "-s", required=True, help="CloudFormation stack name")

    parser.add_argument("--profile", "-p", required=True, help="AWS CLI Profile")

    subparsers = parser.add_subparsers(help="Sub-command help")

    # Subparser for working with S3 buckets
    bucket_parser = subparsers.add_parser("buckets", help="S3 bucket help")

    bucket_parser.add_argument(
        "--exclude-buckets",
        "-e",
        help="Comma-separated list of S3 buckets with data to *retain* (eg. bucket-1,bucket-2,bucket-3)",
    )

    bucket_parser.add_argument("--all", "-a", action="store_true")

    args = parser.parse_args()

    print(args)

    stack_name = args.stack_name

    # init AWS profile & resources
    session = boto3.Session(profile_name=args.profile)
    athena_workgroup = "SD2ReportsWorkgroup"

    cf_client = session.client("cloudformation")
    athena_client = session.client("athena")
    sagemaker_client = session.client("sagemaker")

    # s3_session = session.Session()
    s3 = session.resource(service_name="s3")

    # Exclude specific buckets in a stack from having contents deleted
    if "exclude_buckets" in args and args.exclude_buckets != None:
        exclude_buckets = args.exclude_buckets.split(",")
        s3_buckets_to_clear = get_s3_buckets(args.stack_name, exclude_buckets)
        buckets_cleared = clear_buckets(s3_buckets_to_clear)
    # Remove contents from all buckets in a stack
    elif "all" in args:
        s3_buckets_to_clear = get_s3_buckets(args.stack_name)
        result = clear_buckets(s3_buckets_to_clear)
    # Delete Athena workgroup, the A2I workflow,
    # the contents of all buckets, and the stack
    else:
        try:
            print(f"Deleting Athena workgroup {athena_workgroup}")
            response = athena_client.delete_work_group(
                WorkGroup=athena_workgroup, RecursiveDeleteOption=True
            )
        except Exception as e:
            print(f"Error: could not remove Athena workgroup {athena_workgroup}: {str(e)}")

        try:
            workflow_name = get_a2i_workflow_name(args.stack_name)
            print(f"Deleting A2i workflow {workflow_name}")
            response = sagemaker_client.delete_flow_definition(FlowDefinitionName=workflow_name)
        except Exception as e:
            print(f"Error: could not remove A2i workflow {workflow_name}")

        s3_buckets = get_s3_buckets(args.stack_name)
        buckets_cleared = clear_buckets(s3_buckets)

        if buckets_cleared:
            stack_cleanup = cleanup_stack(args.stack_name)

            if stack_cleanup:
                print(f"Success: stack {args.stack_name} has been removed")
            else:
                print(f"Error: stack {args.stack_name} has not been removed")

        else:
            print(
                f"Could not remove stack {args.stack_name}. One or more of the S3 buckets had data that could not be removed."
            )
            print(
                f"Please remove the data and attempt to delete the stack {args.stack_name} again."
            )
            print(f"In addition, please remember to remove the Athena workgroup {athena_workgroup}")
            print(f"and the A2I workflow")
