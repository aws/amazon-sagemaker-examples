
class ModelPackageArnProvider:
    
    @staticmethod
    def get_gpt2_model_package_arn(current_region):
        mapping = {
    
            "us-east-1" :    "arn:aws:sagemaker:us-east-1:865070037744:model-package/gpt-2-1584040650-de7f6ab78d68d7fdf5f4f39a559d05ac",
            "us-east-2" :  "arn:aws:sagemaker:us-east-1:865070037744:model-package/gpt-2-1584040650-de7f6ab78d68d7fdf5f4f39a559d05ac",
            "us-west-1" :  "arn:aws:sagemaker:us-east-1:865070037744:model-package/gpt-2-1584040650-de7f6ab78d68d7fdf5f4f39a559d05ac",
            "us-west-2" :  "arn:aws:sagemaker:us-east-1:865070037744:model-package/gpt-2-1584040650-de7f6ab78d68d7fdf5f4f39a559d05ac",
            "ca-central-1" :  "arn:aws:sagemaker:us-east-1:865070037744:model-package/gpt-2-1584040650-de7f6ab78d68d7fdf5f4f39a559d05ac",
            "eu-central-1" :  "arn:aws:sagemaker:us-east-1:865070037744:model-package/gpt-2-1584040650-de7f6ab78d68d7fdf5f4f39a559d05ac",
            "eu-west-1" :  "arn:aws:sagemaker:us-east-1:865070037744:model-package/gpt-2-1584040650-de7f6ab78d68d7fdf5f4f39a559d05ac",
            "eu-west-2" :  "arn:aws:sagemaker:us-east-1:865070037744:model-package/gpt-2-1584040650-de7f6ab78d68d7fdf5f4f39a559d05ac",
            "eu-west-3" :  "arn:aws:sagemaker:us-east-1:865070037744:model-package/gpt-2-1584040650-de7f6ab78d68d7fdf5f4f39a559d05ac",
            "eu-north-1" :  "arn:aws:sagemaker:us-east-1:865070037744:model-package/gpt-2-1584040650-de7f6ab78d68d7fdf5f4f39a559d05ac",
            "ap-southeast-1" :  "arn:aws:sagemaker:us-east-1:865070037744:model-package/gpt-2-1584040650-de7f6ab78d68d7fdf5f4f39a559d05ac",
            "ap-southeast-2" :  "arn:aws:sagemaker:us-east-1:865070037744:model-package/gpt-2-1584040650-de7f6ab78d68d7fdf5f4f39a559d05ac",
            "ap-northeast-2" :  "arn:aws:sagemaker:us-east-1:865070037744:model-package/gpt-2-1584040650-de7f6ab78d68d7fdf5f4f39a559d05ac",
            "ap-northeast-1" :  "arn:aws:sagemaker:us-east-1:865070037744:model-package/gpt-2-1584040650-de7f6ab78d68d7fdf5f4f39a559d05ac",
            "ap-south-1" :  "arn:aws:sagemaker:us-east-1:865070037744:model-package/gpt-2-1584040650-de7f6ab78d68d7fdf5f4f39a559d05ac",
            "sa-east-1" :   "arn:aws:sagemaker:us-east-1:865070037744:model-package/gpt-2-1584040650-de7f6ab78d68d7fdf5f4f39a559d05ac"

                    }
        return mapping[current_region]