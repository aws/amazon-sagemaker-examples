class ModelPackageArnProvider:
    @staticmethod
    def get_yolov3_model_package_arn(current_region):
        mapping = {
            "sa-east-1": "arn:aws:sagemaker:sa-east-1:270155090741:model-package/gluoncv-yolo3-darknet531547760-bdf604d6d9c12bf6194b6ae534a638b2",
            "ap-south-1": "arn:aws:sagemaker:ap-south-1:077584701553:model-package/gluoncv-yolo3-darknet531547760-bdf604d6d9c12bf6194b6ae534a638b2",
            "ap-northeast-2": "arn:aws:sagemaker:ap-northeast-2:745090734665:model-package/gluoncv-yolo3-darknet531547760-bdf604d6d9c12bf6194b6ae534a638b2",
            "ap-southeast-1": "arn:aws:sagemaker:ap-southeast-1:192199979996:model-package/gluoncv-yolo3-darknet531547760-bdf604d6d9c12bf6194b6ae534a638b2",
            "ap-southeast-2": "arn:aws:sagemaker:ap-southeast-2:666831318237:model-package/gluoncv-yolo3-darknet531547760-bdf604d6d9c12bf6194b6ae534a638b2",
            "ap-northeast-1": "arn:aws:sagemaker:ap-northeast-1:977537786026:model-package/gluoncv-yolo3-darknet531547760-bdf604d6d9c12bf6194b6ae534a638b2",
            "ca-central-1": "arn:aws:sagemaker:ca-central-1:470592106596:model-package/gluoncv-yolo3-darknet531547760-bdf604d6d9c12bf6194b6ae534a638b2",
            "eu-central-1": "arn:aws:sagemaker:eu-central-1:446921602837:model-package/gluoncv-yolo3-darknet531547760-bdf604d6d9c12bf6194b6ae534a638b2",
            "eu-west-1": "arn:aws:sagemaker:eu-west-1:985815980388:model-package/gluoncv-yolo3-darknet531547760-bdf604d6d9c12bf6194b6ae534a638b2",
            "eu-west-2": "arn:aws:sagemaker:eu-west-2:856760150666:model-package/gluoncv-yolo3-darknet531547760-bdf604d6d9c12bf6194b6ae534a638b2",
            "eu-west-3": "arn:aws:sagemaker:eu-west-3:843114510376:model-package/gluoncv-yolo3-darknet531547760-bdf604d6d9c12bf6194b6ae534a638b2",
            "eu-north-1": "arn:aws:sagemaker:eu-north-1:136758871317:model-package/gluoncv-yolo3-darknet531547760-bdf604d6d9c12bf6194b6ae534a638b2",
            "us-east-1": "arn:aws:sagemaker:us-east-1:865070037744:model-package/gluoncv-yolo3-darknet531547760-bdf604d6d9c12bf6194b6ae534a638b2",
            "us-east-2": "arn:aws:sagemaker:us-east-2:057799348421:model-package/gluoncv-yolo3-darknet531547760-bdf604d6d9c12bf6194b6ae534a638b2",
            "us-west-1": "arn:aws:sagemaker:us-west-1:382657785993:model-package/gluoncv-yolo3-darknet531547760-bdf604d6d9c12bf6194b6ae534a638b2",
            "us-west-2": "arn:aws:sagemaker:us-west-2:594846645681:model-package/gluoncv-yolo3-darknet531547760-bdf604d6d9c12bf6194b6ae534a638b2",
        }
        return mapping[current_region]

    @staticmethod
    def get_ssd_model_package_arn(current_region):
        mapping = {
            "sa-east-1": "arn:aws:sagemaker:sa-east-1:270155090741:model-package/gluoncv-ssd-resnet501547760463-0f9e6796d2438a1d64bb9b15aac57bc0",
            "ap-south-1": "arn:aws:sagemaker:ap-south-1:077584701553:model-package/gluoncv-ssd-resnet501547760463-0f9e6796d2438a1d64bb9b15aac57bc0",
            "ap-northeast-2": "arn:aws:sagemaker:ap-northeast-2:745090734665:model-package/gluoncv-ssd-resnet501547760463-0f9e6796d2438a1d64bb9b15aac57bc0",
            "ap-southeast-1": "arn:aws:sagemaker:ap-southeast-1:192199979996:model-package/gluoncv-ssd-resnet501547760463-0f9e6796d2438a1d64bb9b15aac57bc0",
            "ap-southeast-2": "arn:aws:sagemaker:ap-southeast-2:666831318237:model-package/gluoncv-ssd-resnet501547760463-0f9e6796d2438a1d64bb9b15aac57bc0",
            "ap-northeast-1": "arn:aws:sagemaker:ap-northeast-1:977537786026:model-package/gluoncv-ssd-resnet501547760463-0f9e6796d2438a1d64bb9b15aac57bc0",
            "ca-central-1": "arn:aws:sagemaker:ca-central-1:470592106596:model-package/gluoncv-ssd-resnet501547760463-0f9e6796d2438a1d64bb9b15aac57bc0",
            "eu-central-1": "arn:aws:sagemaker:eu-central-1:446921602837:model-package/gluoncv-ssd-resnet501547760463-0f9e6796d2438a1d64bb9b15aac57bc0",
            "eu-west-1": "arn:aws:sagemaker:eu-west-1:985815980388:model-package/gluoncv-ssd-resnet501547760463-0f9e6796d2438a1d64bb9b15aac57bc0",
            "eu-west-2": "arn:aws:sagemaker:eu-west-2:856760150666:model-package/gluoncv-ssd-resnet501547760463-0f9e6796d2438a1d64bb9b15aac57bc0",
            "eu-west-3": "arn:aws:sagemaker:eu-west-3:843114510376:model-package/gluoncv-ssd-resnet501547760463-0f9e6796d2438a1d64bb9b15aac57bc0",
            "eu-north-1": "arn:aws:sagemaker:eu-north-1:136758871317:model-package/gluoncv-ssd-resnet501547760463-0f9e6796d2438a1d64bb9b15aac57bc0",
            "us-east-1": "arn:aws:sagemaker:us-east-1:865070037744:model-package/gluoncv-ssd-resnet501547760463-0f9e6796d2438a1d64bb9b15aac57bc0",
            "us-east-2": "arn:aws:sagemaker:us-east-2:057799348421:model-package/gluoncv-ssd-resnet501547760463-0f9e6796d2438a1d64bb9b15aac57bc0",
            "us-west-1": "arn:aws:sagemaker:us-west-1:382657785993:model-package/gluoncv-ssd-resnet501547760463-0f9e6796d2438a1d64bb9b15aac57bc0",
            "us-west-2": "arn:aws:sagemaker:us-west-2:594846645681:model-package/gluoncv-ssd-resnet501547760463-0f9e6796d2438a1d64bb9b15aac57bc0",
        }
        return mapping[current_region]
