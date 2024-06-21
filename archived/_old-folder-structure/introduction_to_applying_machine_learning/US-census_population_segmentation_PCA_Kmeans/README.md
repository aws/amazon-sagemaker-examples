## Analyze US census data for population segmentation using Amazon SageMaker

https://aws.amazon.com/blogs/machine-learning/analyze-us-census-data-for-population-segmentation-using-amazon-sagemaker/

*Introduction*

In the United States, with the 2018 midterm elections approaching, people are looking for more information about the voting process. This example notebook explores how we can apply machine learning (ML) to better integrate science into the task of understanding the electorate.

Typically for machine learning applications, clear use cases are derived from labelled data. For example, based on the attributes of a device, such as its age or model number, we can predict its likelihood of failure. We call this supervised learning because there is supervision or guidance towards predicting specific outcomes.

However, in the real world, there are often large data sets where there is no particular outcome to predict, where clean labels are hard to define. It can be difficult to pinpoint exactly what the right outcome is to predict. This type of use case is often exploratory. It seeks to understand the makeup of a dataset and what natural patterns exist. This type of use case is known as unsupervised learning. One example of this is trying to group similar individuals together based on a set of attributes.

The use case this blog post explores is population segmentation. We have taken publicly available, anonymized data from the US census on demographics by different US counties: https://factfinder.census.gov/faces/nav/jsf/pages/index.xhtml. (Note that this product uses the Census Bureau Data API but is not endorsed or certified by the Census Bureau.) The outcome of this analysis are natural groupings of similar counties in a transformed feature space. The cluster that a county belongs to can be leveraged to plan an election campaign, for example, to understand how to reach a group of similar counties by highlighting messages that resonate with that group. More generally, this technique can be applied by businesses in customer or user segmentation to create targeted marketing campaigns. This type of analysis has the ability to uncover similarities that may not be obvious at face value- such as counties CA-Fresno and AZ- Yuma County being grouped together. While intuitively they differ in commonly-examined attributes such as population size and racial makeup, they are more similar than different when viewed along axes such as the mix of employment type.

You can follow along in this sample notebook where you can run the code and interact with the data while reading through the blog post (link is shown above).

There are two goals for this exercise:

1. Walk through a data science workflow using Amazon SageMaker for unsupervised learning using PCA and Kmeans modelling techniques.

2. Demonstrate how users can access the underlying models that are built within Amazon SageMaker to extract useful model attributes. Often, it can be difficult to draw conclusions from unsupervised learning, so being able to access the models for PCA and Kmeans becomes even more important beyond simply generating predictions using the model.

