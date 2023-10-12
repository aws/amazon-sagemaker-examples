# SageMaker Inference Components and Managed Instance Scaling 
For SageMaker Real Time Hosting there are a few new constructs that we are introducing on top of the usual SageMaker endpoint interaction. These include the introduction of inference components and managed instance scaling. SageMaker componenets can be created using an existing SageMaker model, or you can create one by specifying the model artifacts and the container during the creation of an inference endpoint. In addition to being able to scale out copies of your inference components, SageMaker also allows you to set managed instance scaling on your endpoint. SageMaker will scale up or down the number of instances in your endpoint to coorelate to the needs that youre inference components require. 

## Entities/Terminology

Within Sagemaker Hosting there are three main entities:
- Endpoint Config
- Endpoint
- Inference Component

With the entities above we will walk through a 3 part process in a total of 5 notebooks. 
1. Create endpoint 
    1_create_endpoint.ipynb
2. Deploy models to your endpoint using inference components. 
    2a_codegen25_FT_7b.ipynb
    2b_flant5_xxl-tgi.ipynb
    3c_meta-llama2-7b-lmi-autoscaling.ipynb
3. Misc functions and clean up
    3_misc_cleanup.upynb

The notebooks have been organized to be run in sequential order. 
