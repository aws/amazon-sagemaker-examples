### Curating AWS Marketplace Model Package listing and Sample notebook

It is important to make it easy for users of different levels of expertise to use your ML model package from AWS Marketplace. To do so, we recommend you to reuse the instructions and code available as part of [Sample Notebook template](#Sample-Notebook-template) to create a sample notebook that executes end-to-end seamlessly without expecting any user input except ModelPackageArn and an AWS Marketplace subscription to your ML model Package.

For each listing, we recommend you to create a separate custom sample notebook as soon as you have published your listing in AWS Marketplace and host it in a separate public code repository. 

To create a custom sample notebook using [Sample Notebook template](#Sample-Notebook-template), follow these instructions:
In <title_of_your_product>-Model.ipynb, make following changes:
    1. Rename the <title_of_your_product>-Model.ipynb file to something appropriate. Please keep the file extension as ipynb, i.e. a notebook document.
    2. Open the notebook document and specify appropriate title and add overview of the ML Model.
    3. Look for "[Title_of_your_ML Model](Provide link to your marketplace listing of your product)", and update the link to your marketplace listing (four different occurances).
    4. Update imports, several inputs in sections 1 and 2 and start writing code required to demonstrate real-time and batch inference.
    5. The sample notebook template has multiple placeholders which you need to update/remove once you have a working notebook.
    6. Please keep the data/input, data/output folder structure as adviced and please provide multiple input/output files to demonstrate your ML model for different scenarios that demonstrate different features of your ML model.
    7. Write necessary code and ensure that your sample notebook executes without asking user to provide any input except for ModelPackageArn and an AWS Marketplace subscription to your model listing. Once you are satisfied with the sample notebook, remove the hardcoding from model_package_arn.
    8. Ensure that model_package_arn  has a placeholder with a value '<Customer to specify Model package ARN corresponding to their AWS region>'
    9. Ensure that you have removed all "For Seller " notes in red color from your sample notebook. 
    10. Review the table of contents of your notebook and ensure that all links work.
    
Once ready, host your sample notebook on your public GitHUB/Bitbucket/Gitlab repository and link the repository with your AWS Marketplace listing, under additional resources section. Ensure that repository is accessible to public.

    
Next step is to curate your AWS Markeplace listing. For more details, please see the guidance and an example for curating a good AWS Marketplce Model Package listing in "Model Package Listing - Checklist.xlsx" Excel sheet.
