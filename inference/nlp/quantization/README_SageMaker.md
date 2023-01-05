# Pre-requsites
1. Linux shell terminal with AWS CLI installed. 
2. AWS account with access to EC2 (C6i instance type) instance creation.
2. SageMaker access to deploy a SageMaker model, endpoint-configuration, endpoint.
3. IAM access to configure IAM role and policy.
4. Access to Elastic Container Registry (ECR).
5. SageMaker access to create a Notebook with instructions to launch an endpoint.


# Steps Involved

## Role for SageMaker to access S3 and execution rights:
### AWS IAM Role
- Documentation link: https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html
- AmazonSageMaker-ExecutionRole
- AmazonSageMaker-ExecutionPolicy S3 access
```
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::*"
            ]
        }
    ]
}
```

## Launch EC2 instance to generate the model, docker image and model artifcats

1. Launch an EC2 instance with C6i instance type and Ubuntu 20.04 linux AMI. Link to the CloudFormation template here.
1. SSH into the EC2 C6i instance. (All the steps after this are to be executed from the EC2 instance.) 
1. Git clone the sagemaker samples repo.
``` git clone https://github.com/aws-samples/amazon-sagemaker-custom-container.git ```
1. Install Docker. (Learn more about using AWS CLI for ECR, docker here https://docs.aws.amazon.com/AmazonECR/latest/userguide/getting-started-cli.html)
```sudo apt install docker.io  ```
1. Install AWS CLI
``` sudo apt install awscli ```
1. Execute command to create the docker image that will run the quantization and generate a model. The script will create the image and push it to ECR as well. (Same image will be used for the Inference endpoint, as it contains the Python Flask config).
```build_and_push.sh <docker image name> ```
1. Run the docker image as follows and execute the following: 
``` sudo docker run --rm -it -p 8080:8080 -p 8081:8081 -p 8082:8082 -p 7070:7070 -p 7071:7071 bert_large_c6i ```
1. Inside the docker container, execute the following python script to generate both FP32 and INT8 models. (This process may take few minutes depending on the C6i instance size)
``` python quantize_with_ds_ep.py ``` 
1. The above script generate model artifacts required for SageMaker endpoint. In this step we will create and upload the model tar gzip file into AWS S3 bucket.

``` tar -czf both_bert_model.tar.gz model_int8.pt model_fp32.pt tokenizer.json vocab.txt special_tokens_map.json tokenizer_config.json ```

``` aws s3 cp both_bert_model.tar.gz s3://intelc6i ```

## Create SageMaker model and endpoint
1. Register the model in Sagemaker: 
```
aws sagemaker create-model --model-name c6ibothbert --execution-role-arn "<ROLE ARN>" --primary-container '{
  "ContainerHostname": "BertMainContainer1Hostname",
  "Image": "<AWS ACCOUNT NUMBER>.dkr.ecr.us-east-1.amazonaws.com/bert_dataset_flask:latest",
  "ModelDataUrl": "https://intelc6i.s3.amazonaws.com/bert_dataset_model.tar.gz",
  "Environment" : {
         "SAGEMAKER_CONTAINER_LOG_LEVEL": "20",
         "SAGEMAKER_REGION": "us-east-1"
         }
 }' 
``` 
1. Register the inference endpoint configuration for the endpoint referencing the model name and ec2 instance class.
```
aws sagemaker create-endpoint-config --endpoint-config-name c6ibothbert --production-variants '[
{
  "VariantName" : "TestVariant1",
  "ModelName" : "c6ibothbert",
  "InitialInstanceCount" : 1,
  "InstanceType" : "ml.c6i.4xlarge"
 }
]'
```
1. Register the endpoint referencing the endpoint config name.
```
 aws sagemaker create-endpoint --endpoint-name c6ibothbert --endpoint-config-name c6ibothbert 
```
1. Check the status of the endpoint to confirm that it is InService. 
1. To test the model, run the following script:
``` python invoke.py ```
1. Validate the response and check the answer. 
 

Remember to shutdown and cleanup the AWS services and artifacts created as part of this exercise.
Make sure that you save any code or artifacts that you may want for future reference, before you run the cleanup steps. 

## Steps for cleanup
1. Use the CloudFormation console to delete the EC2 instance stack.
2. Run the following commands to delete the SageMaker model and endpoint. 
``` 
aws sagemaker delete-endpoint --endpoint-name  c6ibothbert
aws sagemaker delete-endpoint-config --endpoing-config-name c6ibothbert 
aws sagemaker delete-model --model-name c6ibothbert
```
3. Delete the S3 objects
```

```
4. Delete the docker image from ECR
```
aws ecr batch-delete-image \
      --repository-name <image name> \
      --image-ids imageTag=<tag> \
      --region <AWS region>
```
