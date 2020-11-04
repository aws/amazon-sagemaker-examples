# SageMaker Inference Pipeline using Custom Containers

This demo shows how to use sagemaker container pipelines to manipulate the inference before and after an out of the box SageMaker algorithm, and how to integrate with external services such as DynamoDB as part of that data manipulation. The preprocessing and postprocesing containers in this demo are custom, built using custom training algorithms and a nginx inference endpoint. In this demo we generate some sample data that represents key words in a question that a user might ask a customer support agent and the category of that question as a label. The first step of the pipeline will encode the label (the question category), and transform the list of words with a CategoricalEncoder. The second step is to use xgBoost multi-class predictor to guess the correct label. The final step of the pipeline uses the model from the first step to reverse the label encoding, then look up available agents with a specialty in that category from DynamoDB, and return that data as part of the final prediction.

Prerequisites:

 1. Ensure that the AWS cli has been installed: `https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html`
 2. Ensure that docker has been installed: `https://docs.docker.com/get-docker/`
 3. Have an AWS test account available where you don't mind creating AWS resources including: IAM roles and policies, a DynamoDB table, a Sagemaker notebook, Sagemaker endpoints and training jobs, and Elastic Container Registry respositories.
 4. Configure the AWS cli to use the AWS test account `https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html`

Setup instructions

 1. Run the cloudformation script. As a convenience you can run `npm run-script create`, which will run the Cloudformation commands for you. If you wish to that first update `package.json`, and replace the sections which read `<CF BUCKET HERE>` with the name of a S3 bucket which you want to use to store the packaged cloudformation templates. If you want to run this manually execute the steps in `./scripts/create-stack.sh`.
 2. Load the preprocessor custom container into ECR. First change into the proper path: `cd ./containers/preprocessor/scripts` and run `sh build_and_push.sh $AWS_ACCOUNT_ID us-west-2 custompipeline/preprocessor`
 3. Load the postprocessor custom container into ECR. First change into the proper path: `cd ./containers/preprocessor/scripts` and run `sh build_and_push.sh $AWS_ACCOUNT_ID us-west-2 custompipeline/postprocessor`
 4. Launch the Jupypter notebook. Click on: `restart and run all cells`

Note that this assumes that the stack is running in Oregon (us-west-2). If you wish to run this in a different region, update the `aws_region=us-west-2` reference in `containers/postprocessor/docker/code/predictor.py`
