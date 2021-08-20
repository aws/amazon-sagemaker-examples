## Using Shutterstock's image datasets to train a multi-label image classification model

Welcome! This directory contains the Amazon SageMaker notebook code used in the blog post [**Using Shutterstock's image datasets to train your computer vision models**](https://aws.amazon.com/blogs/awsmarketplace/using-shutterstocks-image-datasets-to-train-your-computer-vision-models/).


For this example, we use the Free Sample: Images & Metadata of “Whole Foods” Shoppers dataset from Shutterstock’s Image Datasets to demonstrate how to train a multi-label image classification model using Shutterstock’s prelabeled image assets. This dataset can be found in the [AWS Data Exchange](https://aws.amazon.com/data-exchange/) and contains images of Whole Foods shoppers. Each image is tagged with 7-50 keywords describing what is seen in the image. 


You can get started using this notebook by following the steps outlined in [this blog](https://aws.amazon.com/blogs/awsmarketplace/using-shutterstocks-image-datasets-to-train-your-computer-vision-models/). 

At a high level, setup involves these 4 steps:

1. Subscribe to the Free Sample: Images & Metadata of “Whole Foods” Shoppers dataset from Shutterstock’s Image Datasets [found here](https://aws.amazon.com/marketplace/pp/prodview-y6xuddt42fmbu?qid=1623195111604&sr=0-1&ref_=srh_res_product_title#offers). Export this dataset to an S3 bucket.
2. Create an Amazon SageMaker Notebook instance [here](https://console.aws.amazon.com/sagemaker/home?region=us-east-1#/notebook-instances). For the development of this notebook, we used an `ml.t2.medium`. Make sure the SageMaker role has access to your Shutterstock Image Dataset S3 bucket. Note that charges apply.
3. Once your notebook instance is ready, click **Open Jupyter** and navigate to the **Sagemaker Examples** tab. Scroll down to find the **AWS Marketplace** section and select it to expand and view the notebooks available.
4. Locate the notebook named `image-classification-multilabel-with-shutterstock-image-datasets.ipynb`, select **Use**, and then select **Create copy** to copy the notebook into your environment.
5. Locate the `TO DO` items to fill in the name of the S3 bucket that is being used to store your image datasets. Also be sure to enter any prefixes (if applicable) that the images are stored under.
6. Walk through the notebook step-by-step and select **Run** to run each cell of code.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

