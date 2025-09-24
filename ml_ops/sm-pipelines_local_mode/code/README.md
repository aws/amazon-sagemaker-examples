## buildspec.sh

This script allows you to create the source artifact to use for SageMaker Training Jobs or SageMaker Processing Jobs

Parameters:
* ALGORITHM_NAME: Mandatory - Name of the algorithm you want to package
* S3_BUCKET_NAME: Optional - S3 bucket name where the package will be uploaded in the path s3://<S3_BUCKET_NAME>/artifact/<ALGORITHM_NAME>

```
.buildspec.sh <ALGORITHM_NAME> <ACCOUNT_ID> <S3_BUCKET_NAME> <KMS_ALIAS>
```

### Example:

#### Processing

```
./buildspec.sh processing test-bucket
```

#### Training

```
./buildspec.sh training test-bucket
```
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
./build_image.sh training torch-1.12.1 latest Dockerfile linux/amd64

