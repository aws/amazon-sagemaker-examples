import argparse
import csv

import boto3

parser = argparse.ArgumentParser(description="Load DynamoDB data")
parser.add_argument("table", type=str, help="DynamoDB table to load the data into")
parser.add_argument(
    "--filename", type=str, default="ddb-data.csv", help="CSV file with DDB sample data"
)
parser.add_argument("--debug", type=bool, default=False, help="Whether to print debug messages")
args = parser.parse_args()

ddb_client = boto3.client("dynamodb")

with open(args.filename) as csvfile:
    reader = csv.DictReader(csvfile)
    request = {args.table: []}
    for row in reader:
        put = {
            "PutRequest": {
                "Item": {
                    "Specialty": {"S": row["Specialty (S)"]},
                    "ID": {"S": row["ID (S)"]},
                    "FirstName": {"S": row["FirstName (S)"]},
                    "LastName": {"S": row["LastName (S)"]},
                    "Available": {"BOOL": row["Available (BOOL)"] == "true"},
                }
            }
        }
        request[args.table].append(put)
    if args.debug:
        print("Batch ddb put request: ", request)
    ddb_client.batch_write_item(RequestItems=request)
