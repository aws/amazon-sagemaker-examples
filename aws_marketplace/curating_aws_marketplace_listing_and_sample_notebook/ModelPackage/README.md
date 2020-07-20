### Curating AWS Marketplace Model Package listing and Sample notebook

It is important that you make it easy for users with different levels of experience to use your ML model package in AWS Marketplace. To do so, we recommend you to reuse the instructions and code available as part of [Sample_Notebook_template](#Sample_Notebook_template) to create a sample notebook that executes end-to-end seamlessly without expecting any user input except `model_package_arn` variable and an AWS Marketplace subscription to your ML model Package.

For each listing, we recommend you to create a separate custom sample notebook repository as soon as you have published your listing in AWS Marketplace and host it in a separate public code repository. 

To create a custom sample notebook using [Sample_Notebook_template](#Sample_Notebook_template), follow these instructions:
In **title_of_your_product**-Model.ipynb, make following changes:
1. Rename the **title_of_your_product**-Model.ipynb file to something appropriate. Keep the file extension as ipynb, i.e. a notebook document.
2. Open the notebook document in Amazon SageMaker or other Jupyter notebook editor, specify an appropriate title, and add overview of the ML model.
3. Look for "[Title_of_your_ML Model](Provide link to your marketplace listing of your product)", and replace it with a link to your marketplace listing.
4. The sample notebook template has multiple placeholders which you need to update to create a high-quality notebook.
5. Retain the data/input, data/output folder structure, and provide multiple input/output files to demonstrate your ML model for different scenarios that demonstrate different features of your ML model.
6. Ensure that your sample notebook executes without asking user to provide any input except for ModelPackageArn and an AWS Marketplace subscription to your model listing. 
7. Ensure that `model_package_arn`  has a placeholder with a value '<Customer to specify Model package ARN corresponding to their AWS region>'.
8. Ensure that you have removed all notes provided for you in red color from the sample notebook template. 
9. Review the table of contents of your notebook and ensure that all links work.
10. Once ready, host the sample notebook on your public GitHUB/Bitbucket/Gitlab repository and link the repository with your AWS Marketplace listing, under additional resources section. 
11. Ensure that repository is accessible to public.
12. Next step is to curate your AWS Markeplace listing. Follow the guidance provided for curating a good AWS Marketplce Model Package listing in [curating_good_model_package_listing.md](#curating_good_model_package_listing.md).
