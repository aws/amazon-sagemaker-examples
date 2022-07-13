# Multi-model SageMaker Pipeline with Hyperparamater Tuning and Experiments Template

This project has two (2) components: (1) `container` - custom Docker image with custom Decision Tree  algorithm using scikit-learn with hyperpameter tuning support, and (2) `sagemaker-pipeline` - a SageMaker pipeline that supports two (2) algorithms: XGBoost on SageMaker container and Decision Tree on custom container built from the first component. The pipeline imports the data from an Athena table and is transformed for ML training using SageMaker Data Wrangler. The pipeline also supports SageMaker HyperParameter Tuning and SageMaker Experiments. The best performing model in terms of R2 Score is then registered to the model registry, ready for inference deployment.

## Start here

In this example, we are solving real estate value regression prediction problem using the
dataset obtained from the [StatLib repository](http://lib.stat.cmu.edu/datasets/) that was derived from the 1990 U.S. census, using one row per census block group. The dataset is imported to an Athena table from S3 and the pipeline imports the data from this table. Data Wrangler transforms the data (i.e. one-hot encoding, etc) as the initial step in the pipeline. The pipeline then proceeds with preprocessing, training using Decision Tree and XGBoost algorithms with hyperparameter tuning, evaluation, and registration of the winning model to the registry. Every trial is recorded in SageMaker Experiments. This pipeline is a modified version of the pipeline provided by [MLOps template for model building, training, and deployment](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-projects-templates-sm.html#sagemaker-projects-templates-code-commit).

Prior to running the pipeline, you have to push the Decision Tree custom container to your own Amazon Elastic Container Registry (ECR). This container is a modified version of [Scikit BYO](https://github.com/aws/amazon-sagemaker-examples/tree/main/advanced_functionality/scikit_bring_your_own/container).

You can use the `restate-project.ipynb` notebook to experiment from SageMaker Studio before you are ready to checkin your code.

## Dataset

This dataset was obtained from the [StatLib repository](http://lib.stat.cmu.edu/datasets/) and derived from the 1990 U.S. census, using one row per census block group. A block group is the smallest geographical unit for which the U.S. Census Bureau publishes sample data (a block group typically has a population of 600 to 3,000 people).

The dataset contains the following features:

```    
    longitude
    latitude
    housingMedianAge
    totalRooms
    totalBedrooms
    population
    households
    medianIncome (dropped in the pipeline)
    medianHouseValue (target)
```


## Assumptions and Prerequisites

- S3 bucket `sagemaker-restate-<AWS ACCOUNT ID>` is created and raw data has been uploaded to `s3://sagemaker-restate-<AWS ACCOUNT ID>/raw/california/`.
- SageMaker project is already created. Recommendation is to create a SageMaker project using [SageMaker-provide MLOps template for model building, training, and deployment template](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-projects-templates-sm.html#sagemaker-projects-templates-code-commit).
- Necessary IAM service roles are already created.

## Security

This sample code is not designed for production deployment out-of-the-box, so further security enhancements may need to be added according to your own requirements before pushing to production. Security recommendations include, but are not limited to, the following:
- Use private ECR
- Use a more defined IAM permission for service roles
- Use interface / gateway VPC endpoints to prevent communication traffic from traversing public network
- Use S3 VPC endpoint policy which controls access to specified Amazon S3 buckets only

The following IAM roles are required:

1 - AmazonSageMakerServiceCatalogProductsUseRole-restate with the following managed policies:
- AmazonAthenaFullAccess
- AmazonSageMakerFullAccess

2 - AWSGlueServiceRole-restate with the following managed policies:
- AmazonS3FullAccess
- AWSGlueServiceRole


[restate-project.ipynb](restate-project.ipynb) has been tested in a SageMaker notebook instance that is using a kernel with Python 3.7 installed. This SageMaker notebook instance is attached with an IAM role with the following managed policies:

- AmazonEC2ContainerRegistryFullAccess
- AmazonS3FullAccess
- AWSGlueConsoleSageMakerNotebookFullAccess
- CloudWatchLogsFullAccess
- AWSCodeCommitReadOnly	AWS managed	- this is needed assuming you code is pulled from CodeCommit
- AmazonSageMakerFullAccess

This SageMaker notebook is attached with an IAM role with the following in-line policy:
```
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "iam:CreateRole",
                "iam:AttachRolePolicy"
            ],
            "Resource": "*"
        }
    ]
}
```

## License

This library is licensed under the MIT-0 License. See the LICENSE file.
