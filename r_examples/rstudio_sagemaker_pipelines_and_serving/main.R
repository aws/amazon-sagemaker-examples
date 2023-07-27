# Make sure you have all the necessary packages installed ------------------
library(dplyr)
library(reticulate)
if (!py_module_available("sagemaker-studio-image-build")){py_install("sagemaker-studio-image-build", pip=TRUE)}
library(readr)

sagemaker <- import('sagemaker')
boto3 <- import('boto3')

session <- sagemaker$Session()
bucket <- session$default_bucket()

role_arn <- sagemaker$get_execution_role()
account_id <- boto3$client("sts")$get_caller_identity()$"Account"
region <- boto3$session$Session()$region_name

# --------- Build the containers ---------

local_path <- dirname(rstudioapi::getSourceEditorContext()$path)
system(paste0("cd ", local_path, " ; sm-docker build . --file ./docker/Dockerfile-processing --repository sagemaker-r-processing:1.0"))

system(paste0("cd ", local_path, " ; sm-docker build . --file ./docker/Dockerfile-train-and-deploy --repository sagemaker-r-train-and-deploy:1.0"))

# --------- Get data ---------
data_file <- 's3://sagemaker-sample-files/datasets/tabular/uci_abalone/abalone.csv'
data_string <- sagemaker$s3$S3Downloader$read_file(data_file)
abalone <- read_csv(file = data_string, col_names = FALSE)
names(abalone) <- c('sex', 'length', 'diameter', 'height', 'whole_weight', 'shucked_weight', 'viscera_weight', 'shell_weight', 'rings')
head(abalone)

dir.create(paste0(local_path,"/data"), showWarnings = FALSE)
write_csv(abalone, paste0(local_path,"/data/abalone_data.csv"))


s3_raw_data <- session$upload_data(path = paste0(local_path,"/data/abalone_data.csv"),
                                   bucket = bucket,
                                   key_prefix = 'pipeline-example/data')

# this will be used for testing
abalone_t <- abalone %>%
  mutate(female = as.integer(ifelse(sex == 'F', 1, 0)),
         male = as.integer(ifelse(sex == 'M', 1, 0)),
         infant = as.integer(ifelse(sex == 'I', 1, 0))) %>%
  select(-sex)





# ------ Run pipeline -------
source("pipeline-example/pipeline.R")
my_pipeline <- get_pipeline(input_data_uri=s3_raw_data)

my_pipeline$definition()

upserted <- my_pipeline$upsert(role_arn=role_arn)
execution <- my_pipeline$start()


# --------- Deploy to serverless endpoint.  ------------

# From the approved models in the model registry we select the one most recently created
approved_models <- boto3$client("sagemaker")$list_model_packages(ModelApprovalStatus='Approved', 
                                                                 ModelPackageGroupName='AbaloneRModelPackageGroup',
                                                                 SortBy='CreationTime',
                                                                 SortOrder='Ascending')
model_package_arn <- approved_models[["ModelPackageSummaryList"]][[1]][["ModelPackageArn"]]

model <- sagemaker$ModelPackage(role=role_arn, 
                                model_package_arn=model_package_arn, 
                                sagemaker_session=session)

serverless_config <- sagemaker$serverless$ServerlessInferenceConfig(memory_size_in_mb=1024L, max_concurrency=5L)
model$deploy(serverless_inference_config=serverless_config, endpoint_name="serverless-r-abalone-endpoint")

# prepare data for a test inference
library(jsonlite)
x = list(features=format_csv(abalone_t[1:3,1:11]))
x = toJSON(x)

# test the endpoint
predictor <- sagemaker$predictor$Predictor(endpoint_name="serverless-r-abalone-endpoint", sagemaker_session=session)
predictor$predict(x)


# ----------- Delete endpoint -------------
predictor$delete_endpoint(delete_endpoint_config=TRUE)
