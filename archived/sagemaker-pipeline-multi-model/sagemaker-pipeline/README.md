## Layout of the SageMaker Pipeline

```
|-- codebuild-buildspec.yml
|-- pipelines
|   |-- restate
|   |   |-- dtree_evaluate.py
|   |   |-- xgb_evaluate.py
|   |   |-- __init__.py
|   |   |-- pipeline.py
|   |   `-- preprocess.py
|   |-- get_pipeline_definition.py
|   |-- __init__.py
|   |-- run_pipeline.py
|   |-- _utils.py
|   `-- __version__.py
|-- README.md
|-- LICENSE
|-- setup.cfg
|-- setup.py
|-- tests
|   `-- test_pipelines.py
`-- tox.ini
```

## Start here
This code repository is created based on a pipeline template from a created Project in SageMaker. Please see [MLOps template for model building, training, and deployment](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-projects-templates-sm.html#sagemaker-projects-templates-code-commit).
