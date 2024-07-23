import sys
import subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "sagemaker"])

import argparse
import pathlib
import time

import boto3
import pandas as pd
import sagemaker
from sagemaker.feature_store.feature_group import FeatureGroup

# Parse argument variables passed via the CreateDataset processing step
parser = argparse.ArgumentParser()
parser.add_argument("--claims-feature-group-name", type=str)
parser.add_argument("--customers-feature-group-name", type=str)
parser.add_argument("--athena-database-name", type=str)
parser.add_argument("--claims-table-name", type=str)
parser.add_argument("--customers-table-name", type=str)
parser.add_argument("--bucket-name", type=str)
parser.add_argument("--bucket-prefix", type=str)
parser.add_argument("--region", type=str, default="us-east-2")
args = parser.parse_args()

region = args.region
boto3.setup_default_session(region_name=region)
s3_client = boto3.client("s3")
account_id = boto3.client("sts").get_caller_identity()["Account"]
now = pd.to_datetime("now")

feature_store_session = sagemaker.Session()
claims_feature_group = FeatureGroup(name=args.claims_feature_group_name, sagemaker_session=feature_store_session)
customers_feature_group = FeatureGroup(
    name=args.customers_feature_group_name, sagemaker_session=feature_store_session
)

claims_table_name = (
    claims_feature_group.describe()["OfflineStoreConfig"]["DataCatalogConfig"]["TableName"]
)
customers_table_name = (
    customers_feature_group.describe()["OfflineStoreConfig"]["DataCatalogConfig"]["TableName"]
)
athena_database_name = customers_feature_group.describe()["OfflineStoreConfig"]["DataCatalogConfig"]["Database"]

print(f'claims_table_name: {claims_table_name}')
print(f'customers_table_name: {customers_table_name}')

claims_feature_group_s3_prefix = "/".join(
    claims_feature_group.describe()
    .get("OfflineStoreConfig")
    .get("S3StorageConfig")
    .get("ResolvedOutputS3Uri")
    .split("/")[3:]
)

customers_feature_group_s3_prefix = "/".join(
    customers_feature_group.describe()
    .get("OfflineStoreConfig")
    .get("S3StorageConfig")
    .get("ResolvedOutputS3Uri")
    .split("/")[3:]
)

print(f'claims_feature_group_s3_prefix: {claims_feature_group_s3_prefix}')
print(f'customers_feature_group_s3_prefix: {customers_feature_group_s3_prefix}')

# wait for data to be added to offline feature store
offline_store_contents = None
while offline_store_contents is None:
    claims_objects = s3_client.list_objects(
        Bucket=args.bucket_name, Prefix=claims_feature_group_s3_prefix
    )
    customers_objects = s3_client.list_objects(
        Bucket=args.bucket_name, Prefix=customers_feature_group_s3_prefix
    )
    # objects_in_bucket = s3_client.list_objects(Bucket=args.bucket_name, Prefix=customers_feature_group_s3_prefix)
    if "Contents" in claims_objects and "Contents" in customers_objects:
        num_datasets = len(claims_objects["Contents"]) + len(customers_objects["Contents"])
    else:
        num_datasets = 0

    if num_datasets >= 2:
        offline_store_contents = customers_objects["Contents"]
    else:
        print(
            f"Waiting for data in offline store: {args.bucket_name}/{customers_feature_group_s3_prefix}"
        )
        time.sleep(60)

print("Data available.")

# query athena table
athena = boto3.client("athena", region_name=region)

training_columns = [
    "fraud",
    "incident_severity",
    "num_vehicles_involved",
    "num_injuries",
    "num_witnesses",
    "police_report_available",
    "injury_claim",
    "vehicle_claim",
    "total_claim_amount",
    "incident_month",
    "incident_day",
    "incident_dow",
    "incident_hour",
    "driver_relationship_self",
    "driver_relationship_na",
    "driver_relationship_spouse",
    "driver_relationship_child",
    "driver_relationship_other",
    "incident_type_collision",
    "incident_type_breakin",
    "incident_type_theft",
    "collision_type_front",
    "collision_type_rear",
    "collision_type_side",
    "collision_type_na",
    "authorities_contacted_police",
    "authorities_contacted_none",
    "authorities_contacted_fire",
    "authorities_contacted_ambulance",
    "customer_age",
    "customer_education",
    "months_as_customer",
    "policy_deductable",
    "policy_annual_premium",
    "policy_liability",
    "auto_year",
    "num_claims_past_year",
    "num_insurers_past_5_years",
    "customer_gender_male",
    "customer_gender_female",
    "policy_state_ca",
    "policy_state_wa",
    "policy_state_az",
    "policy_state_or",
    "policy_state_nv",
    "policy_state_id",
]

training_columns_string = ", ".join(f'"{c}"' for c in training_columns)

query_string = f"""
SELECT DISTINCT {training_columns_string}
FROM "{claims_table_name}" claims LEFT JOIN "{customers_table_name}" customers
ON claims.policy_id = customers.policy_id
"""

print(query_string)

query_execution = athena.start_query_execution(
    QueryString=query_string,
    QueryExecutionContext={"Database": athena_database_name},
    ResultConfiguration={"OutputLocation": f"s3://{args.bucket_name}/query_results/"},
)

query_execution_id = query_execution.get("QueryExecutionId")
query_details = athena.get_query_execution(QueryExecutionId=query_execution_id)


# wait for athena query to finish running
query_status = query_details["QueryExecution"]["Status"]["State"]
print(f"Query ID: {query_execution_id}")
while query_status in ["QUEUED", "RUNNING"]:
    print(f"Query status: {query_status}")
    time.sleep(30)
    query_status = athena.get_query_execution(QueryExecutionId=query_execution["QueryExecutionId"])[
        "QueryExecution"
    ]["Status"]["State"]
print(f"status: {query_status}")

query_details = athena.get_query_execution(QueryExecutionId=query_execution_id)
query_result_s3_uri = (
    query_details.get("QueryExecution", {}).get("ResultConfiguration", {}).get("OutputLocation")
)
uri_split = query_result_s3_uri.split("/")
result_key = f"{uri_split[-2]}/{uri_split[-1]}"

# Split query results into train, test sets
s3_client.download_file(
    Bucket=args.bucket_name, Key=result_key, Filename="/opt/ml/processing/result.csv"
)
dataset = pd.read_csv("/opt/ml/processing/result.csv")
dataset = dataset[training_columns]
train = dataset.sample(frac=0.80, random_state=0)
test = dataset.drop(train.index)

# Write train, test splits to output path
train_output_path = pathlib.Path("/opt/ml/processing/output/train")
test_output_path = pathlib.Path("/opt/ml/processing/output/test")
train.to_csv(train_output_path / "train.csv", index=False)
test.to_csv(test_output_path / "test.csv", index=False)
