import json
import boto3

endpoint_name = "causal-endpoint"
json_name = "payload.json"

with open(json_name, "r") as j:
    contents = json.loads(j.read())
    payload = json.dumps(contents)

runtime = boto3.Session().client(
    service_name="sagemaker-runtime", region_name="us-east-1"
)
response = runtime.invoke_endpoint(
    EndpointName=endpoint_name, ContentType="application/json", Body=payload
)

result = json.loads(response["Body"].read().decode())  # decode response

print(result)
