If you are looking to run multi-node, multi-GPU traininig for Mask RCNN on Sagemaker, run the included 'stack-sm' script from a console with AWS CLI access. 

This script will leverage the 'cfn-sm' AWS Cloud Formation script to create the following services:
- SageMaker Notebook Instance
- S3 Bucket
- VPC
- Security Group (needed to allow Horovod communication between Training Nodes)

The IAM user should have FullAccess to the following roles:
- EC2
- IAM
- SageMaker
- CloudFormation
- S3

Once CloudFormation is complete, copy the 'Outputs' for use inside the Jupyter Notebook.  

Then upload the remaining files to the newly created notebook and then open and continue from the 'ngc-tf-mask_rcnn-s3.ipynb' Jupyter Notebook.

Note - you made need to execute 'chmod -R +x ngc-tf-mask_rcnn-s3' if you receive permission denied after unzipping files to your Notebook.