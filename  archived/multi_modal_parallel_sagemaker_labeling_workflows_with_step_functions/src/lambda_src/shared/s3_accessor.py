import boto3
from shared import log

s3 = boto3.resource("s3")


def uri_to_s3_obj(s3_uri):
    if not s3_uri.startswith("s3://"):
        # This is a local path, indicate using None
        return None
    bucket, key = s3_uri.split("s3://")[1].split("/", 1)
    return s3.Object(bucket, key)


def fetch_s3(s3_uri):
    log.logger.info(f"FETCH {s3_uri}")
    obj = uri_to_s3_obj(s3_uri)
    body = obj.get()["Body"]
    return body.read()


def put_s3(s3_uri, data):
    log.logger.info(f"PUT {s3_uri}")
    obj = uri_to_s3_obj(s3_uri)
    if obj:
        obj.put(Body=data)
        return
    log.logger.info("FAILED TO PUT")
