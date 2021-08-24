## Credit risk prediction and explainability with Amazon SageMaker

This example shows how to user SageMaker Clarify to run explainability jobs on a SageMaker hosted inference pipeline. 

Below is the architecture diagram used in the solution:

![alt text](clarify_inf_pipeline_arch.png)


The notebook performs the following steps:

1. Prepare raw training and test data
2. Create a SageMaker Processing job which performs preprocessing on the raw training data and also produces an SKlearn model which is reused for deployment.
3. Train an XGBoost model on the processed data using SageMaker's built-in XGBoost container
4. Create a SageMaker Inference pipeline containing the SKlearn and XGBoost model in a series
5. Perform inference by supplying raw test data
6. Set up and run explainability job powered by SageMaker Clarify
7. Use open source shap library to create summary and waterfall plots to understand the feature importance better
8. Run bias analysis jobs
9. Clean up


The attached notebook can be run in Amazon SageMaker Studio. 


## Bias and Explainability with Amazon SageMaker Clarify
### Overview
Biases are imbalances in the training data, or the prediction behavior of the model across different groups. Sometimes these biases can cause harms to demographic subgroups, e.g. based age or income bracket. The field of machine learning provides an opportunity to address biases by detecting them and measuring them in your data and model.

Amazon SageMaker Clarify provides machine learning developers with greater visibility into their training data and models so they can identify and limit bias and explain predictions.

We are going to go through each stage of the ML lifecycle, and show where you can include Clarify.

### Problem Formation
In this notebook, we are looking to predict the final grade for a students in a maths class, from the popular Student Performance dataset courtesy of UC Irvine.

For this dataset, final grades range from 0-20, where 15-20 are the most favourable outcomes. This is a multiclass classification problem, where we want to predict which grade a given student will get from 0 to 20.

The benefit of using ML to predict this, is to be able to provide an accurate grade for the student if they aren't able to attend the final exam, due to circumstances outside their control.


