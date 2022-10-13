## build_image.sh

This script allows you to create a custom docker image and push on ECR

Parameters:
* IMAGE_NAME: *Mandatory* - Name of the image you want to build
* REGISTRY_NAME: *Mandatory* - Name of the ECR repository you want to use for pushing the image
* IMAGE_TAG: *Mandatory* - Tag to apply to the ECR image
* DOCKER_FILE: *Mandatory* - Dockerfile to build
* PLATFORM: *Optional* - Target architecture chip where the image is executed
```
./build_image.sh <IMAGE_NAME> <REGISTRY_NAME> <IMAGE_TAG> <DOCKER_FILE> <PLATFORM>
```

Examples:

```
./build_image.sh image_tensorrt nvidia-tensorrt-21.08 latest Dockerfile linux/amd64
```

## create_studio_image.sh

This script allows you to create the Amazon SageMaker Studio Image

Parameters:
* IMAGE_NAME: *Mandatory* - Name of the folder for the image
* REGISTRY_NAME: *Mandatory* - Name of the ECR repository where image is stored
* SM_IMAGE_NAME: *Mandatory* - Name of the image you want to create
* ROLE_ARN: *Mandatory* - Used to get ECR image information when and Image version is created

```
./create_studio_image.sh <IMAGE_NAME> <REGISTRY_NAME> <SM_IMAGE_NAME> <ROLE_ARN>
```

Examples:

```
./create_studio_image.sh image_tensorrt nvidia-tensorrt-21.08 nvidia-tensorrt-21-08 arn:aws:iam::<ACCOUNT_ID>:role/mlops-sagemaker-execution-role
```

## update_studio_image.sh

This script allows you to create the Amazon SageMaker Studio Image

Parameters:
* IMAGE_NAME: *Mandatory* - Name of the folder for the image
* REGISTRY_NAME: *Mandatory* - Name of the ECR repository where image is stored
* SM_IMAGE_NAME: *Mandatory* - Name of the image you want to create
* ROLE_ARN: *Mandatory* - Used to get ECR image information when and Image version is created

```
./update_studio_image.sh <IMAGE_NAME> <REGISTRY_NAME> <SM_IMAGE_NAME> <ROLE_ARN> <AWS_PROFILE_NAME>
```

Examples:

```
./update_studio_image.sh image_tensorrt nvidia-tensorrt-21.08 nvidia-tensorrt-21-08 arn:aws:iam::<ACCOUNT_ID>:role/mlops-sagemaker-execution-role
```

## update_studio_domain.sh

This script allows you to create the Amazon SageMaker Studio Image

```
./update_studio_domain.sh
```