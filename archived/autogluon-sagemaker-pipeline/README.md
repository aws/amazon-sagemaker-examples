# AutoGluon Tabular with SageMaker Pipelines
This example shows how to use [AutoGluon](https://github.com/awslabs/autogluon) with SageMaker Pipelines. To create a SageMaker pipeline, please follow the steps in tutorials such as [this Amazon SageMaker Pipelines tutorial](https://aws.amazon.com/sagemaker/pipelines/). After the pipeline creation is complete, come back here to do modifications for Autogluon. You need to add below files to the official repo:

- ag-config.yaml : expected to be in relevant s3 bucket
- ag_model.py
- tabular_train.py

Beside the above files, you will need to modify `pipeline.py` as shown in this repo.

## Layout of the SageMaker ModelBuild Project Template

The template provides a starting point for bringing your SageMaker Pipeline development to production.

```
|-- codebuild-buildspec.yml
|-- CONTRIBUTING.md
|-- config
|   |--ag-config.yaml
|-- pipelines
|   |-- abalone
|   |   |-- evaluate.py
|   |   |-- __init__.py
|   |   |-- pipeline.py
|   |   |-- preprocess.py
|   |-- get_pipeline_definition.py
|   |-- __init__.py
|   |-- run_pipeline.py
|   |-- _utils.py
|   |-- __version__.py
|   |-- ag_model.py
|-- scripts
|   |-- tabular_train.py
|-- README.md
|-- sagemaker-pipelines-project.ipynb
|-- setup.cfg
|-- setup.py
|-- tests
|   -- test_pipelines.py
|-- tox.ini
```

## Start here
This is a sample code repository that demonstrates how you can organize your code for an ML business solution. This code repository is created as part of creating a Project in SageMaker. 

In this example, we are solving the abalone age prediction problem using the abalone dataset (see below for more on the dataset). The following section provides an overview of how the code is organized and what you need to modify. In particular, `pipelines/pipelines.py` contains the core of the business logic for this problem. It has the code to express the ML steps involved in generating an ML model. You will also find the code for that supports preprocessing and evaluation steps in `preprocess.py` and `evaluate.py` files respectively.

Once you understand the code structure described below, you can inspect the code and you can start customizing it for your own business case. This is only sample code, and you own this repository for your business use case. Please go ahead, modify the files, commit them and see the changes kick off the SageMaker pipelines in the CICD system.

You can also use the `sagemaker-pipelines-project.ipynb` notebook to experiment from SageMaker Studio before you are ready to checkin your code.

A description of some of the artifacts is provided below:
<br/><br/>
Your codebuild execution instructions. This file contains the instructions needed to kick off an execution of the SageMaker Pipeline in the CICD system (via CodePipeline). You will see that this file has the fields definined for naming the Pipeline, ModelPackageGroup etc. You can customize them as required.

```
|-- codebuild-buildspec.yml
```

<br/><br/>
Your pipeline artifacts, which includes a pipeline module defining the required `get_pipeline` method that returns an instance of a SageMaker pipeline, a preprocessing script that is used in feature engineering, and a model evaluation script to measure the Mean Squared Error of the model that's trained by the pipeline. This is the core business logic, and if you want to create your own folder, you can do so, and implement the `get_pipeline` interface as illustrated here.

```
|-- pipelines
|   |-- abalone
|   |   |-- evaluate.py
|   |   |-- __init__.py
|   |   |-- pipeline.py
|   |   `-- preprocess.py

```
<br/><br/>
Utility modules for getting pipeline definition jsons and running pipelines (you do not typically need to modify these):

```
|-- pipelines
|   |-- get_pipeline_definition.py
|   |-- __init__.py
|   |-- run_pipeline.py
|   |-- _utils.py
|   `-- __version__.py
```
<br/><br/>
Python package artifacts:
```
|-- setup.cfg
|-- setup.py
```
<br/><br/>
A stubbed testing module for testing your pipeline as you develop:
```
|-- tests
|   `-- test_pipelines.py
```
<br/><br/>
The `tox` testing framework configuration:
```
`-- tox.ini
```

## Dataset for the Example Abalone Pipeline

The dataset used is the [UCI Machine Learning Abalone Dataset](https://archive.ics.uci.edu/ml/datasets/abalone) [1]. The aim for this task is to determine the age of an abalone (a kind of shellfish) from its physical measurements. At the core, it's a regression problem. 
    
The dataset contains several features - length (longest shell measurement), diameter (diameter perpendicular to length), height (height with meat in the shell), whole_weight (weight of whole abalone), shucked_weight (weight of meat), viscera_weight (gut weight after bleeding), shell_weight (weight after being dried), sex ('M', 'F', 'I' where 'I' is Infant), as well as rings (integer).

[1] Dua, D. and Graff, C. (2019). [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml). Irvine, CA: University of California, School of Information and Computer Science.
