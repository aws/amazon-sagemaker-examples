class ModelPackageArnProvider:
    @staticmethod
    def get_model_package_arn(current_region):
        mapping = {
            "us-east-1": "arn:aws:sagemaker:us-east-1:865070037744:model-package/planning-to-buy-house-basic-28fcb3ca751705854a7171b255d8ef43",
            "us-east-2": "arn:aws:sagemaker:us-east-2:057799348421:model-package/planning-to-buy-house-basic-28fcb3ca751705854a7171b255d8ef43",
            "us-west-1": "arn:aws:sagemaker:us-west-1:382657785993:model-package/planning-to-buy-house-basic-28fcb3ca751705854a7171b255d8ef43",
            "us-west-2": "arn:aws:sagemaker:us-west-2:594846645681:model-package/planning-to-buy-house-basic-28fcb3ca751705854a7171b255d8ef43",
            "ca-central-1": "arn:aws:sagemaker:ca-central-1:470592106596:model-package/planning-to-buy-house-basic-28fcb3ca751705854a7171b255d8ef43",
            "eu-central-1": "arn:aws:sagemaker:eu-central-1:446921602837:model-package/planning-to-buy-house-basic-28fcb3ca751705854a7171b255d8ef43",
            "eu-west-1": "arn:aws:sagemaker:eu-west-1:985815980388:model-package/planning-to-buy-house-basic-28fcb3ca751705854a7171b255d8ef43",
            "eu-west-2": "arn:aws:sagemaker:eu-west-2:856760150666:model-package/planning-to-buy-house-basic-28fcb3ca751705854a7171b255d8ef43",
            "eu-west-3": "arn:aws:sagemaker:eu-west-3:843114510376:model-package/planning-to-buy-house-basic-28fcb3ca751705854a7171b255d8ef43",
            "eu-north-1": "arn:aws:sagemaker:eu-north-1:136758871317:model-package/planning-to-buy-house-basic-28fcb3ca751705854a7171b255d8ef43",
            "ap-southeast-1": "arn:aws:sagemaker:ap-southeast-1:192199979996:model-package/planning-to-buy-house-basic-28fcb3ca751705854a7171b255d8ef43",
            "ap-southeast-2": "arn:aws:sagemaker:ap-southeast-2:666831318237:model-package/planning-to-buy-house-basic-28fcb3ca751705854a7171b255d8ef43",
            "ap-northeast-2": "arn:aws:sagemaker:ap-northeast-2:745090734665:model-package/planning-to-buy-house-basic-28fcb3ca751705854a7171b255d8ef43",
            "ap-northeast-1": "arn:aws:sagemaker:ap-northeast-1:977537786026:model-package/planning-to-buy-house-basic-28fcb3ca751705854a7171b255d8ef43",
            "ap-south-1": "arn:aws:sagemaker:ap-south-1:077584701553:model-package/planning-to-buy-house-basic-28fcb3ca751705854a7171b255d8ef43",
            "sa-east-1": "arn:aws:sagemaker:sa-east-1:270155090741:model-package/planning-to-buy-house-basic-28fcb3ca751705854a7171b255d8ef43",
        }
        return mapping[current_region]
