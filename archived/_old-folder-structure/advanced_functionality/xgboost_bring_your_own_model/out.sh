c4b301c88fc1:amazon-sagemaker-examples julkroll$ git diff
diff --git a/advanced_functionality/xgboost_bring_your_own_model/xgboost_bring_your_own_model.ipynb b/advanced_functionality/xgboost_bring_your_own_model/xgboost_bring_your_own_model.ipynb
index 8df40914..f9b5a511 100644
--- a/advanced_functionality/xgboost_bring_your_own_model/xgboost_bring_your_own_model.ipynb
+++ b/advanced_functionality/xgboost_bring_your_own_model/xgboost_bring_your_own_model.ipynb
@@ -1,18 +1,7 @@
 {
  "cells": [
-  {
-   "cell_type": "code",
-   "execution_count": null,
-   "id": "canadian-powell",
-   "metadata": {},
-   "outputs": [],
-   "source": [
-    "!pip install -Uq xgboost"
-   ]
-  },
   {
    "cell_type": "markdown",
-   "id": "animal-static",
    "metadata": {},
    "source": [
     "# Amazon SageMaker XGBoost Bring Your Own Model\n",
@@ -54,7 +43,6 @@
   {
    "cell_type": "code",
    "execution_count": null,
-   "id": "comic-jonathan",
    "metadata": {},
    "outputs": [],
    "source": [
@@ -77,7 +65,6 @@
   {
    "cell_type": "code",
    "execution_count": null,
-   "id": "revolutionary-egypt",
    "metadata": {},
    "outputs": [],
    "source": [
@@ -88,30 +75,15 @@
   },
   {
    "cell_type": "markdown",
-   "id": "second-traffic",
    "metadata": {},
    "source": [
     "## Optionally, train a scikit learn XGBoost model\n",
     "\n",
-    "These steps are optional and are needed to generate the scikit-learn model that will eventually be hosted using the SageMaker Algorithm contained. \n",
-    "\n",
-    "### Install XGboost\n",
-    "Note that for conda based installation, you'll need to change the Notebook kernel to the environment with conda and Python3. "
-   ]
-  },
-  {
-   "cell_type": "code",
-   "execution_count": null,
-   "id": "expanded-dress",
-   "metadata": {},
-   "outputs": [],
-   "source": [
-    "!conda install -y -c conda-forge xgboost==0.90"
+    "These steps are optional and are needed to generate the scikit-learn model that will eventually be hosted using the SageMaker Algorithm contained. \n"
    ]
   },
   {
    "cell_type": "markdown",
-   "id": "little-still",
    "metadata": {},
    "source": [
     "### Fetch the dataset"
@@ -120,7 +92,6 @@
   {
    "cell_type": "code",
    "execution_count": null,
-   "id": "injured-crawford",
    "metadata": {},
    "outputs": [],
    "source": [
@@ -129,7 +100,6 @@
   },
   {
    "cell_type": "markdown",
-   "id": "tough-facial",
    "metadata": {},
    "source": [
     "### Prepare the dataset for training"
@@ -138,7 +108,6 @@
   {
    "cell_type": "code",
    "execution_count": null,
-   "id": "bright-powder",
    "metadata": {},
    "outputs": [],
    "source": [
@@ -162,7 +131,6 @@
   {
    "cell_type": "code",
    "execution_count": null,
-   "id": "wooden-thesis",
    "metadata": {},
    "outputs": [],
    "source": [
@@ -178,7 +146,6 @@
   },
   {
    "cell_type": "markdown",
-   "id": "strong-height",
    "metadata": {},
    "source": [
     "### Train the XGBClassifier"
@@ -187,7 +154,6 @@
   {
    "cell_type": "code",
    "execution_count": null,
-   "id": "sought-genome",
    "metadata": {},
    "outputs": [],
    "source": [
@@ -202,7 +168,6 @@
   },
   {
    "cell_type": "markdown",
-   "id": "patient-endorsement",
    "metadata": {},
    "source": [
     "### Save the trained model file\n",
@@ -212,7 +177,6 @@
   {
    "cell_type": "code",
    "execution_count": null,
-   "id": "arctic-retail",
    "metadata": {},
    "outputs": [],
    "source": [
@@ -223,7 +187,6 @@
   {
    "cell_type": "code",
    "execution_count": null,
-   "id": "fatty-chapel",
    "metadata": {},
    "outputs": [],
    "source": [
@@ -232,7 +195,6 @@
   },
   {
    "cell_type": "markdown",
-   "id": "forced-illustration",
    "metadata": {},
    "source": [
     "## Upload the pre-trained model to S3"
@@ -241,7 +203,6 @@
   {
    "cell_type": "code",
    "execution_count": null,
-   "id": "molecular-admission",
    "metadata": {},
    "outputs": [],
    "source": [
@@ -252,7 +213,6 @@
   },
   {
    "cell_type": "markdown",
-   "id": "willing-miami",
    "metadata": {},
    "source": [
     "## Set up hosting for the model\n",
@@ -264,19 +224,17 @@
   {
    "cell_type": "code",
    "execution_count": null,
-   "id": "juvenile-glossary",
    "metadata": {},
    "outputs": [],
    "source": [
-    "from sagemaker.amazon.amazon_estimator import get_image_uri\n",
+    "from sagemaker import image_uris\n",
     "\n",
-    "container = get_image_uri(boto3.Session().region_name, \"xgboost\", \"0.90-2\")"
+    "container = image_uris.retrieve(region=boto3.Session().region_name, framework=\"xgboost\", version=\"0.90-2\")"
    ]
   },
   {
    "cell_type": "code",
    "execution_count": null,
-   "id": "competitive-mozambique",
    "metadata": {},
    "outputs": [],
    "source": [
@@ -303,7 +261,6 @@
   },
   {
    "cell_type": "markdown",
-   "id": "announced-affect",
    "metadata": {},
    "source": [
     "### Create endpoint configuration\n",
@@ -314,7 +271,6 @@
   {
    "cell_type": "code",
    "execution_count": null,
-   "id": "alike-experience",
    "metadata": {},
    "outputs": [],
    "source": [
@@ -340,7 +296,6 @@
   },
   {
    "cell_type": "markdown",
-   "id": "otherwise-wiring",
    "metadata": {},
    "source": [
     "### Create endpoint\n",
@@ -350,7 +305,6 @@
   {
    "cell_type": "code",
    "execution_count": null,
-   "id": "experienced-makeup",
    "metadata": {},
    "outputs": [],
    "source": [
@@ -380,7 +334,6 @@
   },
   {
    "cell_type": "markdown",
-   "id": "specific-sheffield",
    "metadata": {},
    "source": [
     "## Validate the model for use\n",
@@ -390,7 +343,6 @@
   {
    "cell_type": "code",
    "execution_count": null,
-   "id": "pediatric-subject",
    "metadata": {},
    "outputs": [],
    "source": [
@@ -399,7 +351,6 @@
   },
   {
    "cell_type": "markdown",
-   "id": "saving-ghost",
    "metadata": {},
    "source": [
     "Lets generate the prediction for a single datapoint. We'll pick one from the test data generated earlier."
@@ -408,7 +359,6 @@
   {
    "cell_type": "code",
    "execution_count": null,
-   "id": "polish-laugh",
    "metadata": {},
    "outputs": [],
    "source": [
@@ -423,7 +373,6 @@
   {
    "cell_type": "code",
    "execution_count": null,
-   "id": "reported-coalition",
    "metadata": {},
    "outputs": [],
    "source": [
@@ -447,7 +396,6 @@
   },
   {
    "cell_type": "markdown",
-   "id": "pursuant-cemetery",
    "metadata": {},
    "source": [
     "### Post process the output\n",
@@ -457,7 +405,6 @@
   {
    "cell_type": "code",
    "execution_count": null,
-   "id": "broken-individual",
    "metadata": {},
    "outputs": [],
    "source": [
@@ -469,7 +416,6 @@
   },
   {
    "cell_type": "markdown",
-   "id": "going-popularity",
    "metadata": {},
    "source": [
     "### (Optional) Delete the Endpoint\n",
@@ -480,7 +426,6 @@
   {
    "cell_type": "code",
    "execution_count": null,
-   "id": "foster-steps",
    "metadata": {},
    "outputs": [],
    "source": [
@@ -490,10 +435,11 @@
  ],
  "metadata": {
   "anaconda-cloud": {},
+  "instance_type": "ml.t3.medium",
   "kernelspec": {
-   "display_name": "Environment (conda_anaconda3)",
+   "display_name": "Python 3 (Data Science)",
    "language": "python",
-   "name": "conda_anaconda3"
+   "name": "python3__SAGEMAKER_INTERNAL__arn:aws:sagemaker:us-west-2:236514542706:image/datascience-1.0"
   },
   "language_info": {
    "codemirror_mode": {
@@ -504,10 +450,11 @@
    "mimetype": "text/x-python",
    "name": "python",
    "nbconvert_exporter": "python",
-   "pygments_lexer": "ipython3"
+   "pygments_lexer": "ipython3",
+   "version": "3.6.13"
   },
   "notice": "Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.  Licensed under the Apache License, Version 2.0 (the \"License\"). You may not use this file except in compliance with the License. A copy of the License is located at http://aws.amazon.com/apache2.0/ or in the \"license\" file accompanying this file. This file is distributed on an \"AS IS\" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License."
  },
  "nbformat": 4,
  "nbformat_minor": 5
-}
\ No newline at end of file
+}
c4b301c88fc1:amazon-sagemaker-examples julkroll$ 
