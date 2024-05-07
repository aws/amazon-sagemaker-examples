# Stable Diffusion XL Fine-Tuning with Kohya SS

*This solution creates all the necessary components to get you started quickly with fine-tuning Stable Diffusion XL with a custom dataset, using a custom training container that leverages Kohya SS to do the fine-tuning. Stable Diffusion allows you to generate images from text prompts. The training is coordinated with a SageMaker pipeline and a SageMaker Training job. This solution automates many of the tedious tasks you must do to set up the necessary infrastructure to run your training. You will use the "kohya-ss-fine-tuning" Notebook to set up the solution. But first, get familiar with the solution components described in this README, which are labeled with their default resource names from the Cloudformation template.*

![Architecture Diagram](kohya-ss-fine-tuning.jpg)

*Prerequisites:*
1. SageMaker Domain configured (to be used with SageMaker Studio).
2. Add the [required permissions](https://aws-blogs-artifacts-public.s3.amazonaws.com/artifacts/ML-16550/sagemaker-policy.json) to the SageMaker Execution Role for your domain.
3. SageMaker Domain User Profile configured.
4. If you will run the Cloudformation template via the console, the proper IAM permissions must be assigned to your user.

*Follow these steps to get started:*

1. Navigate to Amazon SageMaker Studio in your AWS account. Run your JupyterLab space.
2. Click on "Terminal".
3. You will check out just the required directories of the SageMaker Examples git repository next (so you don't have to download the entire repo). Run the following commands from the terminal. If successful, you should see the output "Your branch is up to date with 'origin/main'".

        git clone --no-checkout https://github.com/azograby/amazon-sagemaker-examples.git
        # TODO: change to official aws examples repo once personal repo PR is approved and merged. PR: https://github.com/aws/amazon-sagemaker-examples/pull/4481
        # git clone --no-checkout https://github.com/aws/amazon-sagemaker-examples.git
        cd amazon-sagemaker-examples/
        git sparse-checkout set use-cases/text-to-image-fine-tuning
        git checkout

4. In Amazon SageMaker Studio, in the left-hand navigation pane, click the File Browser and navigate to the project directory (amazon-sagemaker-examples/use-cases/text-to-image-fine-tuning). Open the Jupyter Notebook named "kohya-ss-fine-tuning.ipynb".
5. The default runtime kernel is set to use Python 3 automatically. You now have a kernel that is ready to run commands. You may now continue with this Notebook to start setting up your solution.

---
---
  
## **Solution Components:**

### S3 Bucket (sagemaker-kohya-ss-fine-tuning-\<accountid\>)
The S3 bucket where we will upload the custom dataset (images and captions) to fine-tune a custom model. We will upload the images and the kohya ss configuration file in the notebook, and the SageMaker pipeline will orchestrate the training and output the model to this same S3 bucket.

### CodeCommit Repository (kohya-ss-fine-tuning-container-image)
The CodeCommit source code repository that contains the code to build the training container image (Dockerfile), the training code itself (train), and the build spec that will be used by CodeBuild to create the docker image (buildspec.yml). Upon changes to these repository files, an EventBridge rule is invoked to build a new container image via CodeBuild, which then pushes the new image to the ECR repository.

### ECR Repository (kohya-ss-fine-tuning)
The ECR repository for the training container image. A container image will be built and pushed to this repository, containing the [Kohya SS](https://github.com/bmaltais/kohya_ss.git) program, which will be used to train a custom SDXL model.

### CodeBuild Project (kohya-ss-fine-tuning-build-container)
The CodeBuild project that builds the training container image and pushes the image to ECR. The environment variables in the template.yml for the project can be modified to change the Kohya SS branch version. The GitHub repository (https://github.com/bmaltais/kohya_ss.git) has been tested as of version v22.1.1. If you use newer versions, you will want to check the Dockerfile and the docker-compose.yaml file in the repository, and the training entrypoint for SDXL (sdxl_train_network.py) in the custom "train" file located in this repository to see if any modifications need to be made.

### EventBridge Rule (kohya-ss-fine-tuning-trigger-new-image-build-rule)
Updating the CodeCommit repository code will trigger the CodeBuild project that builds a new training container image and pushes it to ECR. This does NOT kick off a training job.

### SageMaker Pipeline (kohya-ss-fine-tuning-pipeline)
The SageMaker pipeline that orchestrates training the custom model. Currently it only contains a single step for training, but is meant to be extended with additional steps if required. It takes the custom training image from ECR, and the dataset/config located in S3, and initiates a SageMaker training job. Once completed, it outputs the model to the same S3 bucket. You may then take this packaged model and run inference against it. For inference, you may build another custom container for inference, or use tools such as [Automatic1111 Stable Diffusion Web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) and leverage the model that was just created. This pipeline must be executed manually, given specific input parameters which you may override.

### IAM Roles
IAM roles are created for the SageMaker Pipeline execution (custom-sagemaker-pipeline-execution-role), SageMaker service role (custom-sagemaker-service-role), CodeBuild service role (custom-codebuild-service-role), and EventBridge role (custom-build-new-training-container-image-rule-role).