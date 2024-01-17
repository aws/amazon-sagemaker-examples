# Automated Model Approval Pipeline with SageMaker Pipelines, Cloudwatch Events and AWS Lambda

This SageMaker example showcases how you can trigger a SageMaker Pipeline when a new model is registered into the Model Registry.

## Prerequisites

The pipeline requires a registered a model to be evaluated and approved.

## Contents

[sm_pipeline_automated-approval.ipynb](sm_pipeline_automated-approval.ipynb): Notebook explaining the pipeline step-by-step.

[model-approval-checks.py](model-approval-checks.py): Script used in `RegisteredModelValidationStep` step in pipeline to retrieve the artifacts associated with a model and compare
values of interest with specified thresholds (quality, bias, explainability). Returns a dictionary of boolean values for each check.

[validate-model.py](validate-model.py): Script used in `UpdateModelStatusStep` step in pipeline to update the model status to `Approved` or `Rejected` based on the dictionary returned by the previous step.

[./lambda-trigger/lambda-function.py](./lambda-trigger/lambda-function.py): Lambda function used to trigger the SageMaker Pipeline based on an EventBridge event.

[./lambda-trigger/eventbridge-event-patern.json](./lambda-trigger/eventbridge-event-patern.json): EventBridge event pattern used to trigger the Lambda function. 