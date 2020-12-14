## Build, Automate, Manage, and Scale ML Workflows using Amazon SageMaker Pipelines

We recently announced Amazon SageMaker Pipelines (https://aws.amazon.com/sagemaker/pipelines/), the first purpose-built, easy-to-use Continuous Integration and Continuous Delivery (CI/CD) service for machine learning. SageMaker Pipelines has three main components which improves the operational resilience and reproducibility of your workflows: pipelines, model registry, and projects. 

The main components of Amazon SageMaker Pipelines are shown in the diagram below. SageMaker Projects includes MLOps templates that automatically provision the underlying resources needed to enable CI/CD capabilities for your Machine Learning Development Lifecycle (MLDC). Customers can use a number of built-in templates or create your own custom templates. SageMaker Pipelines can be used independently to create automated workflows;however, when used in combination with SageMaker Projects the additional CI/CD capabilities are provide automatically. As the diagram below illustrates, SageMaker’s built-in MLOps templates can automatically provision all of the underlying resources to build CI/CD pipelines that tap into AWS Developer Tools (https://aws.amazon.com/products/developer-tools/) and AWS Service Catalog (https://aws.amazon.com/servicecatalog/).

![A typical ML Application pipeline](img/pipeline-full.png)
