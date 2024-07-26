class ModelPackageArnProvider:
    @staticmethod
    def get_gpt2_model_package_arn(current_region):
        mapping = {
            "us-east-1": "arn:aws:sagemaker:us-east-1:865070037744:model-package/gpt-2-1584040650-de7f6ab78d68d7fdf5f4f39a559d05ac",
            "us-east-2": "arn:aws:sagemaker:us-east-2:057799348421:model-package/gpt-2-1584040650-de7f6ab78d68d7fdf5f4f39a559d05ac",
            "us-west-1": "arn:aws:sagemaker:us-west-1:382657785993:model-package/gpt-2-1584040650-de7f6ab78d68d7fdf5f4f39a559d05ac",
            "us-west-2": "arn:aws:sagemaker:us-west-2:594846645681:model-package/gpt-2-1584040650-de7f6ab78d68d7fdf5f4f39a559d05ac",
            "ca-central-1": "arn:aws:sagemaker:ca-central-1:470592106596:model-package/gpt-2-1584040650-de7f6ab78d68d7fdf5f4f39a559d05ac",
            "eu-central-1": "arn:aws:sagemaker:eu-central-1:446921602837:model-package/gpt-2-1584040650-de7f6ab78d68d7fdf5f4f39a559d05ac",
            "eu-west-1": "arn:aws:sagemaker:eu-west-1:985815980388:model-package/gpt-2-1584040650-de7f6ab78d68d7fdf5f4f39a559d05ac",
            "eu-west-2": "arn:aws:sagemaker:eu-west-2:856760150666:model-package/gpt-2-1584040650-de7f6ab78d68d7fdf5f4f39a559d05ac",
            "eu-west-3": "arn:aws:sagemaker:eu-west-3:843114510376:model-package/gpt-2-1584040650-de7f6ab78d68d7fdf5f4f39a559d05ac",
            "eu-north-1": "arn:aws:sagemaker:eu-north-1:136758871317:model-package/gpt-2-1584040650-de7f6ab78d68d7fdf5f4f39a559d05ac",
            "ap-southeast-1": "arn:aws:sagemaker:ap-southeast-1:192199979996:model-package/gpt-2-1584040650-de7f6ab78d68d7fdf5f4f39a559d05ac",
            "ap-southeast-2": "arn:aws:sagemaker:ap-southeast-2:666831318237:model-package/gpt-2-1584040650-de7f6ab78d68d7fdf5f4f39a559d05ac",
            "ap-northeast-2": "arn:aws:sagemaker:ap-northeast-2:745090734665:model-package/gpt-2-1584040650-de7f6ab78d68d7fdf5f4f39a559d05ac",
            "ap-northeast-1": "arn:aws:sagemaker:ap-northeast-1:977537786026:model-package/gpt-2-1584040650-de7f6ab78d68d7fdf5f4f39a559d05ac",
            "ap-south-1": "arn:aws:sagemaker:ap-south-1:077584701553:model-package/gpt-2-1584040650-de7f6ab78d68d7fdf5f4f39a559d05ac",
            "sa-east-1": "arn:aws:sagemaker:sa-east-1:270155090741:model-package/gpt-2-1584040650-de7f6ab78d68d7fdf5f4f39a559d05ac",
        }
        return mapping[current_region]
