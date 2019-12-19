import json
import boto3
from urllib.parse import urlparse
import math
import logging


def lambda_handler(event, context):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.debug("{}".format(event))

    result = PostProcessNERAnnotation().process(event)

    return result


class PostProcessNERAnnotation:

    def __init__(self):
        self.s3_client = None

    @property
    def logger(self):
        return logging.getLogger(__name__)

    @property
    def s3_client(self):
        self.__s3_client__ = self.__s3_client__ or boto3.resource('s3')
        return self.__s3_client__

    @s3_client.setter
    def s3_client(self, value):
        self.__s3_client__ = value

    def process(self, event):

        payload_uri = event["payload"]["s3Uri"]
        labelAttributeName = event["labelAttributeName"]

        bucket_name = urlparse(payload_uri).netloc
        key = urlparse(payload_uri).path.strip("/")

        payload_object = self.s3_client.Object(bucket_name, key)
        payload = payload_object.get()["Body"].read().decode('utf-8')
        self.logger.debug("{}".format(payload))

        result = []
        for r in json.loads(payload):
            annotations_hit = {}
            valid_annotations = []

            # Consolidate annotaions for the same record from various workers..
            # Annotations for various workers for the same record.. Pick the majority ones
            num_workers = len(r["annotations"])
            # threshold atleast 10% of the workers should have identified this
            threshold = math.ceil(num_workers * 10 / 100)

            for a in r["annotations"]:
                entities_annotations = json.loads(a["annotationData"]["content"])

                for key, value in json.loads(entities_annotations["entities"]).items():
                    start_index = value['startindex']
                    token = value['tokentext']
                    length = len(token)
                    hit_key = "{}#{}".format(start_index, length)
                    if hit_key not in annotations_hit: annotations_hit[hit_key] = 0

                    annotations_hit[hit_key] += 1
                    if annotations_hit[hit_key] == threshold:
                        valid_annotations.append({"start_index": start_index, "length": length, "token": token})

            result.append({
                "datasetObjectId": r["datasetObjectId"],
                "consolidatedAnnotation": {
                    "content": {
                        labelAttributeName: {"entities": valid_annotations}
                    }
                }
            })
        return result
