library(jsonlite)
library(reticulate)
library(stringr)


args = commandArgs(trailingOnly=TRUE)
print(args)

boto3 <- import('boto3')
s3 <- boto3$client('s3')

# Setup parameters
# Container directories
prefix <- '/opt/ml'
input_path <- paste(prefix, 'input/data', sep='/')
output_path <- paste(prefix, 'output', sep='/')
model_path <- paste(prefix, 'model', sep='/')
code_dir <- paste(prefix, 'code', sep='/')
inference_code_dir <- paste(model_path, 'code', sep='/')

# This is where the hyperparamters are saved by the estimator on the container instance
param_path <- paste(prefix, 'input/config/hyperparameters.json', sep='/')

# if param file exists then it is a training job, otherwise it's inference
if (file.exists(param_path)) {
  params <- read_json(param_path)
  
  s3_source_code_tar <- gsub('"', '', params$sagemaker_submit_directory)
  script <- gsub('"', '', params$sagemaker_program)
  
  bucketkey <- str_replace(s3_source_code_tar, "s3://", "")
  bucket <- str_remove(bucketkey, "/.*")
  key <- str_remove(bucketkey, ".*?/")
  
  s3$download_file(bucket, key, "sourcedir.tar.gz")
  untar("sourcedir.tar.gz", exdir=code_dir)
  
  source(file.path(code_dir, script))
  #train()
  
} else {
  print("inference time")
  source(file.path(inference_code_dir, "deploy.R"))
}

