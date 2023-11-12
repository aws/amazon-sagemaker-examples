# Stable Diffusion XL Fine-Tuning with Kohya SS

*This solution creates all the necessary components to get you started quickly with fine-tuning Stable Diffusion XL with a custom dataset, using a custom training container that leverages Kohya SS to do the fine-tuning. Stable Diffusion allows you to generate images from text prompts. The training is coordinated with a SageMaker pipeline and a SageMaker Training job. This solution automates many of the tedious tasks you must do to set up the necessary infrastructure to run your training. You will use the "kohya-ss-fine-tuning" Notebook to set up the solution. But first, get familiar with the solution components described in this README, which are labeled with their default resource names from the Cloudformation template.*

*Then, follow these steps to get started:*

1. Navigate to Amazon SageMaker Studio in your AWS account. On the home screen, click "Open Launcher".
2. In the "Utilities and files" section, click "System terminal".
3. You will check out just the required directories of the SageMaker Examples git repository next (so you don't have to download the entire repo). Run the following commands from the System terminal.

        git clone --no-checkout https://github.com/aws/amazon-sagemaker-examples.git
        cd amazon-sagemaker-examples/
        git sparse-checkout set use-cases/text-to-image-fine-tuning
        git checkout
4. In Amazon SageMaker Studio, in the left-hand navigation pane, click the File Browser and navigate to the project directory. Open the Jupyter Notebook named kohya-ss-fine-tuning.ipynb. You may now continue with this Notebook to start setting up your solution.

---

### S3 Bucket (kohya-ss-fine-tuning-\<accountid\>)
The S3 bucket where we will upload the custom dataset (images) to fine-tune a custom model. We will upload the images and the kohya ss configuration file in the notebook, and the SageMaker pipeline will orchestrate the training and output the model to this same S3 bucket.

### CodeCommit Repository (kohya-ss-fine-tuning-container-image)
The CodeCommit source code repository that contains the code to build the training container image (Dockerfile), the training code itself (train), and the build spec that will be used by CodeBuild to create the docker image (buildspec.yml). Upon changes to this repository files, an EventBridge rule is invoked to build a new container image via CodeBuild, which then pushes the new image/tag to the ECR repository.

### ECR Repository (kohya-ss-fine-tuning)
The ECR repository for the training container image. A container image will be built and pushed to this repository, containing the Kohya SS program, which will be used to train a custom SDXL model.

### CodeBuild Project (kohya-ss-fine-tuning-build-container)
The CodeBuild project that builds the training container image and pushes the image to ECR. The environment variables in the template.yml for the project can be modified to change the Kohya SS branch version. The GitHub repository (https://github.com/bmaltais/kohya_ss.git) has been tested as of version v21.8.9. If you use newer versions, you will want to check the Dockerfile and the docker-compose.yaml file in the repository, and the training entrypoint for SDXL (sdxl_train_network.py) in the custom "train" file located in this repository.

### EventBridge Rule (kohya-ss-fine-tuning-trigger-new-image-build-rule)
Updating the CodeCommit repository will trigger the CodeBuild project that builds a new training container image and pushes it to ECR. This does NOT kick off a training job.

### SageMaker Pipeline (kohya-ss-fine-tuning-pipeline)
The SageMaker pipeline that orchestrates training the custom model. Currently it only contains a single step for training, but is meant to be extended with additional steps if required. It takes the custom training image from ECR, and the dataset/config located in S3, and initiates a SageMaker training job. Once completed, it outputs the model to the same S3 bucket. You may then take this packaged model and run inference against it. For inference, you may build another custom container for inference, or use tools such as Automatic1111 and leverage the model that was just created. This pipeline must be executed manually, given specific input parameters which you may override.

### IAM Roles
IAM roles are created for the SageMaker Pipeline execution (custom-sagemaker-pipeline-execution-role), SageMaker service role (custom-sagemaker-service-role), CodeBuild service role (custom-codebuild-service-role), and EventBridge role (custom-build-new-training-container-image-rule-role)