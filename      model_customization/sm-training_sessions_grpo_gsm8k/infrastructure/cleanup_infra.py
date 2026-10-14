"""
Clean up AWS resources created while running through sm-training_sessions_grpo_gsm8k.ipynb, and sm-training_sessions_deploy_grpo_checkpoint.ipynb: 
inference endpoints, training sessions, the Model Package Group and its checkpoints, 
S3 output artifacts, and the IAM role from setup_infra.py.

Region/account/bucket are auto-discovered the same way the notebooks discover
them (`sagemaker.core.helper.session_helper.Session`) -- there's no config
file to read.

Every category is off by default -- running with no flags only *lists* what's
out there. Pass the matching flag to delete that category. Review
what gets listed before opting in; some resources may still be in use.

Usage:
    python cleanup_infra.py                              # list only, deletes nothing
    python cleanup_infra.py --stop-sessions
    python cleanup_infra.py --delete-endpoints my-endpoint-name
    python cleanup_infra.py --delete-s3                   # deletes objects under outputs/training-sessions
    python cleanup_infra.py --delete-model-package-group  # deletes the group + all checkpoints in it
    python cleanup_infra.py --delete-roles                # deletes the IAM role from setup_infra.py
    python cleanup_infra.py --all                         # stop sessions + delete S3 + delete MPG + delete role
                                                           # (endpoints still need --delete-endpoints,
                                                           # since deleting inference endpoints by name
                                                           # is a deliberate, one-at-a-time action)
"""

import argparse
import time

import boto3
from sagemaker.core.helper.session_helper import Session
from sagemaker.train.training_session import ServiceClient

# Must match the MODEL_PACKAGE_GROUP_NAME constants in the notebooks --
# the Model Package Group ARN is rebuilt from this + region + account_id
# rather than read from a shared file.
MODEL_PACKAGE_GROUP_NAMES = [
    "gsm8k-grpo-training-sessions",  # sm-training_sessions_grpo_gsm8k.ipynb
    "contractnli-sft-training-sessions",  # 3-sft.ipynb
    "gsm8k-challenge-training-sessions",  # 4-rlvr.ipynb
]
ROLE_NAME = "SageMakerTrainingSessionRole"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--stop-sessions", action="store_true", help="Stop all running/created training sessions"
)
parser.add_argument(
    "--delete-endpoints",
    nargs="*",
    default=[],
    metavar="NAME",
    help="Delete these inference endpoints (and their inference components/models/configs)",
)
parser.add_argument(
    "--delete-s3",
    action="store_true",
    help="Delete all objects under outputs/training-sessions in the bucket",
)
parser.add_argument(
    "--delete-model-package-group",
    action="store_true",
    help="Delete the Model Package Groups created by the notebooks, and every checkpoint in them",
)
parser.add_argument(
    "--delete-roles", action="store_true", help="Delete the IAM role created by setup_infra.py"
)
parser.add_argument(
    "--all", action="store_true", help="Stop sessions + delete S3 + delete MPG + delete role"
)
args = parser.parse_args()

sess = Session()
REGION = sess.boto_region_name
ACCOUNT_ID = sess.account_id()
BUCKET_NAME = sess.default_bucket()
DEFAULT_PREFIX = sess.default_bucket_prefix
OUTPUT_PREFIX = (
    f"{DEFAULT_PREFIX}/outputs/training-sessions" if DEFAULT_PREFIX else "outputs/training-sessions"
)

sm = boto3.client("sagemaker", region_name=REGION)
s3 = boto3.client("s3", region_name=REGION)
iam = boto3.client("iam")

print(f"Account: {ACCOUNT_ID}")
print(f"Region:  {REGION}")
print(f"Bucket:  {BUCKET_NAME}")
print(f"Model Package Groups: {', '.join(MODEL_PACKAGE_GROUP_NAMES)}")
print("---")


# Step 1: Inference endpoints
print("\n[1/5] Inference endpoints...")
resp = sm.list_endpoints(MaxResults=20, SortBy="CreationTime", SortOrder="Descending")
for ep in resp["Endpoints"]:
    print(f"  {ep['EndpointName']:<40} {ep['EndpointStatus']:<15} {ep['CreationTime']}")
if not resp["Endpoints"]:
    print("  (none found)")

for ep_name in args.delete_endpoints:
    print(f"\n  Cleaning up endpoint: {ep_name}")

    ics = sm.list_inference_components(EndpointNameEquals=ep_name)
    for ic in ics.get("InferenceComponents", []):
        ic_name = ic["InferenceComponentName"]
        try:
            sm.delete_inference_component(InferenceComponentName=ic_name)
            print(f"    Deleted inference component: {ic_name}")
        except Exception as e:
            print(f"    Failed to delete IC {ic_name}: {e}")
    if ics.get("InferenceComponents"):
        time.sleep(5)  # let ICs finish deleting before touching the endpoint

    try:
        ec_name = sm.describe_endpoint(EndpointName=ep_name)["EndpointConfigName"]
    except Exception:
        ec_name = ep_name

    try:
        sm.delete_endpoint(EndpointName=ep_name)
        print(f"    Deleted endpoint: {ep_name}")
    except Exception as e:
        print(f"    Failed to delete endpoint: {e}")

    try:
        sm.delete_endpoint_config(EndpointConfigName=ec_name)
        print(f"    Deleted endpoint config: {ec_name}")
    except Exception as e:
        print(f"    Failed to delete endpoint config: {e}")

    try:
        sm.delete_model(ModelName=ep_name)
        print(f"    Deleted model: {ep_name}")
    except Exception as e:
        print(f"    Failed to delete model: {e}")

if not args.delete_endpoints and resp["Endpoints"]:
    print("\n  Pass --delete-endpoints NAME [NAME ...] to remove any of the above.")


# Step 2: Training sessions
print("\n[2/5] Training sessions...")
# role_arn is now a required ServiceClient parameter (a default for the sessions it creates);
# the list/stop calls below don't use it. Built from ROLE_NAME, so this also
# works after --delete-roles has removed the role.
svc = ServiceClient(region=REGION, role_arn=f"arn:aws:iam::{ACCOUNT_ID}:role/{ROLE_NAME}")
# Only this repo's sessions, by name prefix, so other sessions in the account are left alone.
# list_sessions returns TrainingSessionSummary objects (attributes, not dict keys) on SDK >= 1.19.
sessions = [
    s
    for prefix in ("gsm8k-grpo", "contractnli-sft", "gsm8k-challenge")
    for s in svc.list_sessions(name_contains=prefix)
]

running = []
for s in sessions:
    name = s.training_session_arn.split("/")[-1]
    status = str(s.status).split(".")[-1].capitalize()  # SessionStatus enum -> "Running"
    print(f"  {name:<45} {status}")
    if status in ("Running", "Created", "Pending"):
        running.append(name)
if not sessions:
    print("  (none found)")

if args.stop_sessions or args.all:
    for name in running:
        try:
            svc.stop_session(training_session=name)
            print(f"  Stopped: {name}")
        except Exception as e:
            print(f"  Failed to stop {name}: {e}")
elif running:
    print(f"\n  {len(running)} active session(s). Pass --stop-sessions (or --all) to stop them.")


# Step 3: S3 output artifacts
print("\n[3/5] S3 output artifacts...")
resp = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=OUTPUT_PREFIX, Delimiter="/", MaxKeys=20)
prefixes = [p["Prefix"] for p in resp.get("CommonPrefixes", [])]
print(f"  {len(prefixes)} run prefix(es) under s3://{BUCKET_NAME}/{OUTPUT_PREFIX}*/")
for p in prefixes:
    print(f"    {p}")

if args.delete_s3 or args.all:
    paginator = s3.get_paginator("list_objects_v2")
    count = 0
    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=OUTPUT_PREFIX):
        objects = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
        if objects:
            s3.delete_objects(Bucket=BUCKET_NAME, Delete={"Objects": objects})
            count += len(objects)
    print(f"  Deleted {count} objects from s3://{BUCKET_NAME}/{OUTPUT_PREFIX}*/")
elif prefixes:
    print("\n  Pass --delete-s3 (or --all) to permanently delete these.")


# Step 4: Model Package Groups (created by the notebooks)
print("\n[4/5] Model Package Groups...")
found = {}  # group name -> list of package summaries
for group in MODEL_PACKAGE_GROUP_NAMES:
    try:
        pkgs = sm.list_model_packages(ModelPackageGroupName=group)["ModelPackageSummaryList"]
        found[group] = pkgs
        print(f"  {group}: {len(pkgs)} checkpoint(s)")
    except sm.exceptions.ClientError:
        print(f"  {group}: not found")

if args.delete_model_package_group or args.all:
    for group, pkgs in found.items():
        for pkg in pkgs:
            pkg_arn = pkg["ModelPackageArn"]
            try:
                sm.delete_model_package(ModelPackageName=pkg_arn)
                print(f"    Deleted checkpoint: {pkg_arn}")
            except Exception as e:
                print(f"    Failed to delete {pkg_arn}: {e}")
        try:
            sm.delete_model_package_group(ModelPackageGroupName=group)
            print(f"  Deleted model package group: {group}")
        except Exception as e:
            print(f"  Failed to delete model package group {group}: {e}")
elif any(found.values()):
    print(
        "\n  Pass --delete-model-package-group (or --all) to delete the group and its checkpoints."
    )


# Step 5: IAM role (created by setup_infra.py)
print("\n[5/5] IAM role...")
if args.delete_roles or args.all:
    try:
        for policy in iam.list_role_policies(RoleName=ROLE_NAME)["PolicyNames"]:
            iam.delete_role_policy(RoleName=ROLE_NAME, PolicyName=policy)
            print(f"  Deleted policy {policy} from {ROLE_NAME}")
        iam.delete_role(RoleName=ROLE_NAME)
        print(f"  Deleted role: {ROLE_NAME}")
    except iam.exceptions.NoSuchEntityException:
        print(f"  Role not found: {ROLE_NAME}")
    except Exception as e:
        print(f"  Failed to delete {ROLE_NAME}: {e}")
else:
    print(f"  Role from setup: {ROLE_NAME}")
    print(
        "  Pass --delete-roles (or --all) to remove it. Only do this if you're done with the beta."
    )


did_something = (
    args.stop_sessions
    or args.delete_endpoints
    or args.delete_s3
    or args.delete_model_package_group
    or args.delete_roles
    or args.all
)
print("\n" + "=" * 60)
print("CLEANUP COMPLETE" if did_something else "NOTHING DELETED (list-only run -- see flags above)")
print("=" * 60)
