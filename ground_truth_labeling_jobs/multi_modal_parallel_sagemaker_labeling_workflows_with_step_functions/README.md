## Install SAM

https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/serverless-sam-cli-install.html

## AWS Configuration

- You must have setup an AWS cli tool before deployment (along with correct access keys)
- Setup the AWS cli (see https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html)
  and ensure your credentials are working by running a sample command
  - Ex: `aws s3 ls`


## Deployment

- Ensure you have a private workforce created in SageMaker console, if you use
  SMGT for other labeling jobs, one should already be created
  - https://ap-northeast-1.console.aws.amazon.com/sagemaker/groundtruth#/labeling-workforces
- If not, create one: Labeling workforces -> Private -> Create Private Team -> Team Name
  - Select "Create a private team with AWS Cognito"
  - Team Name: "default" (this team won't be used, terraform will deploy your own teams + send invites)
  - Add a dummy email account
  - Add organization name, and contact email
  - Click create
- Copy ids to use during deployment
  - Copy the `Amazon Cognito user pool`
  - Copy the `App Client`
- cd cloudformation
- `./launch_sam`, provide variables copied from previous steps, note your profile will come from
  your aws CLI tool
  - You can skip manually copying these parameters by setting them as environment variables
  - `AWS_PROFILE`, `COGNITO_POOL_ID`, `COGNITO_POOL_CLIENT_ID`
- After deployment, login to SageMaker console, look for any workteams prefixed with "smgt-workflows",
  invite your personal emails to these workteams to see jobs when in progress

## Operation

### Jupyter
- Install Jupyter notebook: https://jupyter.org/install
- `cd scripts && jupyter notebook .`
- Browser window should open with Jupyter notebook, replace configuration variables
  - Update `AWSRestApiId` using the api id deployed to your account (ID is in the APIGateway console)
  - Update `AWSAccountId` with your account id
  - Update `AWSRegion` with the region you deployed infrastructure into
- See cell comments for all API commands that can be run
- Run the "Batch Create" API call to create a batch with a unique `batchId`

- Login to worker console using invite email sent to email addresses setup in "Deployment Instructions"
- Wait for jobs to show up, submit jobs
- Use the "Batch Show" api with your job's `batchId` to get status information about the batch

# How to extend

## Multi-level step function

- Update `cloudformation/main-packaged.yml` to include new state.
  - Copy "TriggerLabelingFirstLevel" and "CheckForFirstLevelCompletion" to new states, "TriggerLabelingThirdLevel", and "CheckForThirdLevelCompletion"
  - Update all paths within the new states to refer to `third_level` instead of `first_level`, set `jobLevel` to 3
  - Update "TriggerLabelingFirstLevel"'s next state to point to "CheckForThirdLevelCompletion"
  - Update "CheckForSecondLevelCompletion"'s next state to be "TriggerLabelingThirdLevel"
  - Update "CheckForThirdLevelCompletion"'s next to point to terminal state ("CopyLogsAndSendBatchCompleted")
- Update constants for new job level
  - Update "BatchMetadataType" to include new field `THIRD_LEVEL`
  - Update "BatchCurrentStep" to include new field `THIRD_LEVEL`
- Update `src/lambda_src/step_functions_trigger_labeling_job/main.py`'s `lambda_handler` function.
  - Update the conditional setting metadata type based on `job_level`. Include a check for `job_level` 3 and set metadata type to `THIRD_LEVEL`
  - Update `chainable_batches` logic to handle a `job_level` == 3. You can decide here if you want to support chaining from any previous job or only from jobs in the previous level (like second level).
- Update batch/create api: `src/lambda_src/api_batch_create/main.py`
  - Update validation for "jobLevel must be 1 or 2" and add 3 to list of valid `jobLevel`

