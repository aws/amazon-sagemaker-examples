# Scheduling SKLearn with Amazon SageMaker Pipelines

In this notebook, we will use Amazon SageMaker Pipelines to create two workflows with Scikit-Learn. We will create a pipeline that preprocess data and trains a model (we will use scikit-learn Pipeline), then we will schedule inference with SageMaker Batch Transform.