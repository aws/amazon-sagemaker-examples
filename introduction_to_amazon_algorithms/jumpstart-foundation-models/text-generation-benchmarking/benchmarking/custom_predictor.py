import requests
from typing import Optional
from urllib.parse import urlparse
import json
import sagemaker

from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer


class CustomPredictor:
    predictor: Predictor

    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        predictor: Predictor = None,
        instance_type=None,
    ):
        if endpoint_url is None and predictor is None:
            raise ValueError(f"both endpoint_url and predictor are none in CustomPredictor.")
        self.endpoint_url = endpoint_url
        self.predictor = predictor
        self.instance_type = instance_type
        if self.predictor is not None:
            self.endpoint_name = self.predictor.endpoint_name
        else:
            self.endpoint_name = self.endpoint_url

    def predict(self, payload):
        if self.predictor is None:
            response = requests.post(self.endpoint_url, json=payload)
            return response.text
        else:
            return self.predictor.predict(payload, custom_attributes="accept_eula=True")

    def delete_model(self):
        if self.predictor is not None:
            self.predictor.delete_model()

    def delete_endpoint(self):
        if self.predictor is not None:
            self.predictor.delete_endpoint()

    def toJson(self):
        obj_dict = self.__dict__.copy()
        obj_dict.pop(self.predictor, None)
        return json.dumps(obj_dict)
