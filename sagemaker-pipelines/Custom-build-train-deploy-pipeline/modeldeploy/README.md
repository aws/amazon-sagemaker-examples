## MLOps for SageMaker Endpoint Deployment

This is a sample code repository for demonstrating how you can organize your code for deploying an realtime inference Endpoint infrastructure. This code repository is created as part of creating a Project in SageMaker. 

This code repository has the code to find the latest approved ModelPackage for the associated ModelPackageGroup and automaticaly deploy it to the Endpoint on detecting a change (`build.py`). This code repository also defines the CloudFormation template which defines the Endpoints as infrastructure. It also has configuration files associated with `staging` and `prod` stages. 

Upon triggering a deployment, the CodePipeline pipeline will deploy 2 Endpoints - `staging` and `prod`. After the first deployment is completed, the CodePipeline waits for a manual approval step for promotion to the prod stage. You will need to go to CodePipeline AWS Managed Console to complete this step.

You own this code and you can modify this template to change as you need it, add additional tests for your custom validation. 

A description of some of the artifacts is provided below:


## Layout of the SageMaker ModelBuild Project Template

`buildspec.yml`
 - this file is used by the CodePipeline's Build stage to build a CloudFormation template.

`build.py`
 - this python file contains code to get the latest approve package arn and exports staging and configuration files. This is invoked from the Build stage.

`endpoint-config-template.yml`
 - this CloudFormation template file is packaged by the build step in the CodePipeline and is deployed in different stages.

`staging-config.json`
 - this configuration file is used to customize `staging` stage in the pipeline. You can configure the instance type, instance count here.

`prod-config.json`
 - this configuration file is used to customize `prod` stage in the pipeline. You can configure the instance type, instance count here.

`test\buildspec.yml`
  - this file is used by the CodePipeline's `staging` stage to run the test code of the following python file

`test\test.py`
  - this python file contains code to describe and invoke the staging endpoint. You can customize to add more tests here.
