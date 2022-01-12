# Amazon Comprehend with SageMaker Pipelines

This SageMaker example showcases how you can deploy a custom text classification using Amazon Comprehend and SageMaker Pipelines.

## Contents

[sm_pipeline_with_comprehend.ipynb](sm_pipeline_with_comprehend.ipynb): Notebook explaining the pipeline step-by-step.

[prepare_data.py](prepare_data.py): Script used in ComprehendProcess step in pipeline for data preparation used for training and testing.

[train_eval_comprehend.py](train_eval_comprehend.py): Script used in ComprehendTrainAndEval step in pipeline to train and evaluate the Amazon Comprehend model.

[deploy_comprehend.py](deploy_comprehend.py): Script used in ComprehendDeploy step in pipeline to deploy an Amazon Comprehend model endpoint.

[iam_helper.py](iam_helper.py): Helper function to create IAM for lambda function.

[test_comprehend_lambda.py](test_comprehend_lambda.py): Lambda handler used to perform inference using the Amazon Comprehend model endpoint.
