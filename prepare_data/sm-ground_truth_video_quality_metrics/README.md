## Audit and Improve Video Annotation Quality Using Amazon SageMaker Ground Truth

This notebook walks through how to evaluate the quality of video annotations received from SageMaker Ground Truth annotators using several metrics like IoU, Rolling IoU and Embedding Comparisons. In addition, it shows how to flag frames which may not be labeled properly using these quality metrics and send those frames for verification/audit jobs using SageMaker Ground truth. 

Note: The standard functionality of this notebook will work with the standard Conda Python3/Data Science kernel, however there is an optional section that uses a PyTorch model to generate image embeddings. To run that section, please use a Conda PyTorch Python3 kernel.

## Prerequisites

You will create some of the resources you need to launch a Ground Truth audit labeling job in this notebook. You must create the following resources before executing this notebook:

* A work team. A work team is a group of workers that complete labeling tasks. If you want to preview the worker UI and execute the labeling task you will need to create a private work team, add yourself as a worker to this team, and provide the work team ARN below. This [GIF](images/create-workteam-loop.gif) demonstrates how to quickly create a private work team on the Amazon SageMaker console. To learn more about private, vendor, and Amazon Mechanical Turk workforces, see [Create and Manage Workforces
](https://docs.aws.amazon.com/sagemaker/latest/dg/sms-workforce-management.html).