## Credit risk prediction and explainability with Amazon SageMaker

This example shows how to user SageMaker Clarify to run explainability jobs on a SageMaker hosted inference pipelines. 

Below is the architecture diagram used in the solution:

![alt text](clarify_inf_pipeline_arch.png)


The notebook performs the following steps:

1. Prepare raw training and test data
2. Create a SageMakerProcessing job which performs preprocessing on the raw training data and also produce an SKlearn model which is reused while deployment.
3. Train an XGBoost model on the processed data using SageMaker's built-in XGBoost container.
4. Create a SageMaker Inference pipeline containing the SKlearn and XGBoost model in a series.
5. Perform inference by supplying raw test data
6. Set up and run explainability job powered by SageMaker Clarify
7. Use open source shap library to create summary and waterfall plots to understand the feature importance better.
8. Run bias analysis jobs
9. Clean up


The attached notebook can be run in Amazon SageMaker Studio. 


