# Serving JPMML-based Tree-based models on Amazon SageMaker

This example notebook demonstrates how to bring your own java-based container and serve PMML based Models on Amazon SageMaker for Inference.

## Getting Started

- Open AWS Console
- Search “SageMaker” and click “Amazon Sagemaker”
<img width="320" alt="image" src="https://user-images.githubusercontent.com/46861009/165638576-abc2cc6f-a891-4355-b807-40a712087157.png">

- Click Notebook -> Notebook Instances -> Create notebook Instances
<img width="468" alt="image" src="https://user-images.githubusercontent.com/46861009/165638603-92d3f89b-03a3-45f5-995b-9b133263f998.png">

- Create Notebook Instance with the following details

  Notebook instance name – “sagemaker-immersion-day”
  
  Notebook instance type – “ml.m5.xlarge”
  
- Go back to Notebook instance tab page and Open the notebook instance when status is InService -> Click “Open JupyterLab”. This will open a JupyterLab Interface.
- Go to "Git" from the top menu and select "Clone a Repository" as shown below 
<img width="312" alt="Notebook-git-clone" src="https://user-images.githubusercontent.com/46861009/165638266-d5215304-42ce-4e09-a00d-6a4eadf36ec4.png">
- Clone this repo - https://github.com/dhawalkp/sagemaker-byoc-pmml-example.git as shown below

<img width="299" alt="Notebook-git-clone-w-url" src="https://user-images.githubusercontent.com/46861009/165638391-df060167-dac2-4872-9b08-39b49ea15775.png">
  
- Click “sagemaker-byoc-pmml-example” folder -> JPMML_Models_SageMaker.ipynb and follow the instructions 
<img width="448" alt="image" src="https://user-images.githubusercontent.com/46861009/165638469-4e43f585-c762-4c5c-be04-84bd2617f309.png">

- Select conda_python3 as the kernel in order to execute the notebook.




 

