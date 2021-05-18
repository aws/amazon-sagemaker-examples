import json


class InferenceSpecification:

    template = """
{    
    "InferenceSpecification": {
        "Containers" : [{"Image": "IMAGE_REPLACE_ME"}],
        "SupportedTransformInstanceTypes": INSTANCES_REPLACE_ME,
        "SupportedRealtimeInferenceInstanceTypes": INSTANCES_REPLACE_ME,
        "SupportedContentTypes": CONTENT_TYPES_REPLACE_ME,
        "SupportedResponseMIMETypes": RESPONSE_MIME_TYPES_REPLACE_ME
    }
}
"""

    def get_inference_specification_dict(
        self, ecr_image, supports_gpu, supported_content_types=None, supported_mime_types=None
    ):
        return json.loads(
            self.get_inference_specification_json(
                ecr_image, supports_gpu, supported_content_types, supported_mime_types
            )
        )

    def get_inference_specification_json(
        self, ecr_image, supports_gpu, supported_content_types=None, supported_mime_types=None
    ):
        if supported_mime_types is None:
            supported_mime_types = []
        if supported_content_types is None:
            supported_content_types = []
        return (
            self.template.replace("IMAGE_REPLACE_ME", ecr_image)
            .replace("INSTANCES_REPLACE_ME", self.get_supported_instances(supports_gpu))
            .replace("CONTENT_TYPES_REPLACE_ME", json.dumps(supported_content_types))
            .replace("RESPONSE_MIME_TYPES_REPLACE_ME", json.dumps(supported_mime_types))
        )

    def get_supported_instances(self, supports_gpu):
        cpu_list = [
            "ml.m4.xlarge",
            "ml.m4.2xlarge",
            "ml.m4.4xlarge",
            "ml.m4.10xlarge",
            "ml.m4.16xlarge",
            "ml.m5.large",
            "ml.m5.xlarge",
            "ml.m5.2xlarge",
            "ml.m5.4xlarge",
            "ml.m5.12xlarge",
            "ml.m5.24xlarge",
            "ml.c4.xlarge",
            "ml.c4.2xlarge",
            "ml.c4.4xlarge",
            "ml.c4.8xlarge",
            "ml.c5.xlarge",
            "ml.c5.2xlarge",
            "ml.c5.4xlarge",
            "ml.c5.9xlarge",
            "ml.c5.18xlarge",
        ]
        gpu_list = [
            "ml.p2.xlarge",
            "ml.p2.8xlarge",
            "ml.p2.16xlarge",
            "ml.p3.2xlarge",
            "ml.p3.8xlarge",
            "ml.p3.16xlarge",
        ]

        list_to_return = cpu_list

        if supports_gpu:
            list_to_return = cpu_list + gpu_list

        return json.dumps(list_to_return)
