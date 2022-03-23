library(reticulate)

local_path <- dirname(rstudioapi::getSourceEditorContext()$path)

get_pipeline <- function(input_data_uri){
  
  s3_raw_data <- sagemaker$workflow$parameters$ParameterString(
    name="InputData",
    default_value=input_data_uri
  )
  
  sagemaker <- import('sagemaker')
  boto3 <- import('boto3')
  
  session <- sagemaker$Session()
  bucket <- session$default_bucket()
  
  role_arn <- sagemaker$get_execution_role()
  account_id <- boto3$client("sts")$get_caller_identity()$"Account"
  region <- boto3$session$Session()$region_name
  
  ecr_repository_processing = "sagemaker-r-processing"
  tag_processing <- ":1.0"
  
  ecr_repository_training = "sagemaker-r-train-and-deploy"
  tag_training <- ":1.0"
  
  
  # Define preprocessing step  ------------------
  processing_repository_uri <- paste0(account_id,".dkr.ecr.",region,".amazonaws.com","/",ecr_repository_processing,tag_processing)
  
  
  script_processor <- sagemaker$processing$ScriptProcessor(command=list("Rscript"), 
                                                           image_uri=processing_repository_uri,
                                                           role=role_arn,
                                                           instance_count=1L,
                                                           instance_type="ml.m5.large")
  
  output_s3_data_location <- paste0("s3://", bucket, "/pipeline-example/processing_output")
  s3_processing_input <- sagemaker$processing$ProcessingInput(source = s3_raw_data, destination="/opt/ml/processing/input")
  s3_processing_output1 <- sagemaker$processing$ProcessingOutput(output_name="abalone_train", destination=output_s3_data_location, source="/opt/ml/processing/output/train")
  s3_processing_output2 <- sagemaker$processing$ProcessingOutput(output_name="abalone_test",  destination=output_s3_data_location, source="/opt/ml/processing/output/test")
  s3_processing_output3 <- sagemaker$processing$ProcessingOutput(output_name="abalone_valid", destination=output_s3_data_location, source="/opt/ml/processing/output/valid")
  
  step_process <- sagemaker$workflow$steps$ProcessingStep(name="ProcessingStep",
                                                          code=paste0(local_path, "/preprocessing/preprocessing.R"),
                                                          processor=script_processor,
                                                          inputs=list(s3_processing_input),
                                                          outputs=list(s3_processing_output1, s3_processing_output2, s3_processing_output3))
  
  
  # Define training step  ------------------
  model_image_uri <- paste0(account_id,".dkr.ecr.",region,".amazonaws.com","/",ecr_repository_training,tag_training)
  
  train_estimator <- sagemaker$estimator$Estimator(model_image_uri,
                                                   role_arn, 
                                                   source_dir=paste0(local_path,"/training/code/"),
                                                   entry_point="train.R",
                                                   instance_count=1L, 
                                                   instance_type='ml.m5.xlarge',
                                                   output_path=paste0("s3://", bucket, "/pipeline-example/training_output"),
                                                   sagemaker_session=session,
                                                   metric_definitions=list(list(Name="rmse-validation", Regex= "Calculated validation RMSE: ([0-9.]+);.*$")))
  
  step_train <- sagemaker$workflow$steps$TrainingStep(
    name="TrainingStep",
    estimator=train_estimator,
    inputs=list(training = sagemaker$inputs$TrainingInput(
                            s3_data=step_process$properties$ProcessingOutputConfig$Outputs["abalone_train"]$S3Output$S3Uri,
                            content_type="text/csv")
                )
  )
  
  # Evaluate model step -----------------
  
  script_eval <- sagemaker$processing$ScriptProcessor(
    image_uri=processing_repository_uri,
    command=list("Rscript"),
    instance_type="ml.m5.large",
    instance_count=1L,
    base_job_name="script-evaluate",
    role=role_arn
  )
  
  evaluation_report <- sagemaker$workflow$properties$PropertyFile(
    name="EvaluationReport", output_name="evaluation", path="evaluation.json"
  )
  
  step_eval = sagemaker$workflow$steps$ProcessingStep(
    name="EvaluateModel",
    processor=script_eval,
    inputs=list(
      sagemaker$processing$ProcessingInput(
        source=step_train$properties$ModelArtifacts$S3ModelArtifacts,
        destination="/opt/ml/processing/model"
      ),
      sagemaker$processing$ProcessingInput(
        source=step_process$properties$ProcessingOutputConfig$Outputs["abalone_test"]$S3Output$S3Uri,
        destination="/opt/ml/processing/test"
      )
    ),
    outputs=list(
      sagemaker$processing$ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation")
    ),
    code=paste0(local_path, "/postprocessing/evaluation.R"),
    property_files=list(evaluation_report)
  )
  
  
  # Register model ------------------
  
  model_metrics <- sagemaker$model_metrics$ModelMetrics(
    model_statistics=sagemaker$model_metrics$MetricsSource(
      s3_uri=paste0(step_eval$arguments[["ProcessingOutputConfig"]][["Outputs"]][[1]][["S3Output"]][["S3Uri"]], "/evaluation.json"),
      content_type="application/json"
    )
  )
  
  step_register <- sagemaker$workflow$step_collections$RegisterModel(
    name="RegisterModelStep",
    estimator=train_estimator,
    model_data=step_train$properties$ModelArtifacts$S3ModelArtifacts,
    content_types=list("application/json"),
    response_types=list("application/json"),
    inference_instances=list("ml.t2.medium", "ml.m5.xlarge"),
    transform_instances=list("ml.m5.xlarge"),
    model_package_group_name="AbaloneRModelPackageGroup",
    approval_status="Approved",  # we are automatically registering the model as approved but in more general case you'd want to set this to "PendingManualApproval",
    model_metrics=model_metrics
  )
  
  # Condition Step
  
  cond_lte <- sagemaker$workflow$conditions$ConditionLessThanOrEqualTo(
    left=sagemaker$workflow$functions$JsonGet(
      step_name=step_eval$name,
      property_file=evaluation_report,
      json_path="regression_metrics.rmse.value"
    ),
    right=6.0
  )
  
  step_cond <- sagemaker$workflow$condition_step$ConditionStep(
    name="rmseConditional",
    conditions=list(cond_lte),
    if_steps=list(step_register),
    else_steps=list()
  )
  
  
  # Construct the Pipeline
  
  pipeline_name <-"AbalonePipelineUsingR"
  pipeline = sagemaker$workflow$pipeline$Pipeline(
    name=pipeline_name,
    parameters=list(s3_raw_data),
    steps=list(step_process, step_train, step_eval, step_cond)
  )
  
  return(pipeline)
}