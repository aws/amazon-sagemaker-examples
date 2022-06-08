![SageMaker](https://github.com/aws/amazon-sagemaker-examples/raw/main/_static/sagemaker-banner.png)

# Amazon SageMaker Examples

Example Jupyter notebooks that demonstrate how to build, train, and deploy machine learning models using Amazon SageMaker.

## :books: Background

[Amazon SageMaker](https://aws.amazon.com/sagemaker/) is a fully managed service for data science and machine learning (ML) workflows.
You can use Amazon SageMaker to simplify the process of building, training, and deploying ML models.

The [SageMaker example notebooks](https://sagemaker-examples.readthedocs.io/en/latest/) are Jupyter notebooks that demonstrate the usage of Amazon SageMaker.

## :hammer_and_wrench: Setup

The quickest setup to run example notebooks includes:
- An [AWS account](http://docs.aws.amazon.com/sagemaker/latest/dg/gs-account.html)
- Proper [IAM User and Role](http://docs.aws.amazon.com/sagemaker/latest/dg/authentication-and-access-control.html) setup
- An [Amazon SageMaker Notebook Instance](http://docs.aws.amazon.com/sagemaker/latest/dg/gs-setup-working-env.html)
- An [S3 bucket](http://docs.aws.amazon.com/sagemaker/latest/dg/gs-config-permissions.html)

## :computer: Usage

These example notebooks are automatically loaded into SageMaker Notebook Instances.
They can be accessed by clicking on the `SageMaker Examples` tab in Jupyter or the SageMaker logo in JupyterLab.

Although most examples utilize key Amazon SageMaker functionality like distributed, managed training or real-time hosted endpoints, these notebooks can be run outside of Amazon SageMaker Notebook Instances with minimal modification (updating IAM role definition and installing the necessary libraries).

As of February 7, 2022, the default branch is named "main". See our [announcement](https://github.com/aws/amazon-sagemaker-examples/discussions/3131) for details and how to update your existing clone.

## :notebook: Examples

### Introduction to Ground Truth Labeling Jobs

These examples provide quick walkthroughs to get you up and running with the labeling job workflow for Amazon SageMaker Ground Truth.

- [Bring your own model for SageMaker labeling workflows with active learning](ground_truth_labeling_jobs/bring_your_own_model_for_sagemaker_labeling_workflows_with_active_learning) is an end-to-end example that shows how to bring your custom training, inference logic and active learning to the Amazon SageMaker ecosystem.
- [From Unlabeled Data to a Deployed Machine Learning Model: A SageMaker Ground Truth Demonstration for Image Classification](ground_truth_labeling_jobs/from_unlabeled_data_to_deployed_machine_learning_model_ground_truth_demo_image_classification) is an end-to-end example that starts with an unlabeled dataset, labels it using the Ground Truth API, analyzes the results, trains an image classification neural net using the annotated dataset, and finally uses the trained model to perform batch and online inference.
- [Ground Truth Object Detection Tutorial](ground_truth_labeling_jobs/ground_truth_object_detection_tutorial) is a similar end-to-end example but for an object detection task.
- [Basic Data Analysis of an Image Classification Output Manifest](ground_truth_labeling_jobs/data_analysis_of_ground_truth_image_classification_output) presents charts to visualize the number of annotations for each class, differentiating between human annotations and automatic labels (if your job used auto-labeling). It also displays sample images in each class, and creates a pdf which concisely displays the full results.
- [Training a Machine Learning Model Using an Output Manifest](ground_truth_labeling_jobs/object_detection_augmented_manifest_training) introduces the concept of an "augmented manifest" and demonstrates that the output file of a labeling job can be immediately used as the input file to train a SageMaker machine learning model.
- [Annotation Consolidation](ground_truth_labeling_jobs/annotation_consolidation) demonstrates Amazon SageMaker Ground Truth annotation consolidation techniques for image classification for a completed labeling job.

### Introduction to Applying Machine Learning

These examples provide a gentle introduction to machine learning concepts as they are applied in practical use cases across a variety of sectors.

- [Predicting Customer Churn](introduction_to_applying_machine_learning/xgboost_customer_churn) uses customer interaction and service usage data to find those most likely to churn, and then walks through the cost/benefit trade-offs of providing retention incentives. This uses Amazon SageMaker's implementation of [XGBoost](https://github.com/dmlc/xgboost) to create a highly predictive model.
- [Cancer Prediction](introduction_to_applying_machine_learning/breast_cancer_prediction) predicts Breast Cancer based on features derived from images, using SageMaker's Linear Learner.
- [Ensembling](introduction_to_applying_machine_learning/ensemble_modeling) predicts income using two Amazon SageMaker models to show the advantages in ensembling.
- [Video Game Sales](introduction_to_applying_machine_learning/video_game_sales) develops a binary prediction model for the success of video games based on review scores.
- [MXNet Gluon Recommender System](introduction_to_applying_machine_learning/gluon_recommender_system) uses neural network embeddings for non-linear matrix factorization to predict user movie ratings on Amazon digital reviews.
- [Fair Linear Learner](introduction_to_applying_machine_learning/fair_linear_learner) is an example of an effective way to create fair linear models with respect to sensitive features.
- [Population Segmentation of US Census Data using PCA and Kmeans](introduction_to_applying_machine_learning/US-census_population_segmentation_PCA_Kmeans) analyzes US census data and reduces dimensionality using PCA then clusters US counties using KMeans to identify segments of similar counties.
- [Document Embedding using Object2Vec](introduction_to_applying_machine_learning/object2vec_document_embedding) is an example to embed a large collection of documents in a common low-dimensional space, so that the semantic distances between these documents are preserved.
- [Traffic violations forecasting using DeepAR](introduction_to_applying_machine_learning/deepar_chicago_traffic_violations) is an example to use daily traffic violation data to predict pattern and seasonality to use Amazon DeepAR alogorithm.

### SageMaker Automatic Model Tuning

These examples introduce SageMaker's hyperparameter tuning functionality which helps deliver the best possible predictions by running a large number of training jobs to determine which hyperparameter values are the most impactful.

- [XGBoost Tuning](hyperparameter_tuning/xgboost_direct_marketing) shows how to use SageMaker hyperparameter tuning to improve your model fit.
- [BlazingText Tuning](hyperparameter_tuning/blazingtext_text_classification_20_newsgroups) shows how to use SageMaker hyperparameter tuning with the BlazingText built-in algorithm and 20_newsgroups dataset..
- [TensorFlow Tuning](hyperparameter_tuning/tensorflow_mnist) shows how to use SageMaker hyperparameter tuning with the pre-built TensorFlow container and MNIST dataset.
- [MXNet Tuning](hyperparameter_tuning/mxnet_mnist) shows how to use SageMaker hyperparameter tuning with the pre-built MXNet container and MNIST dataset.
- [HuggingFace Tuning](hyperparameter_tuning/huggingface_multiclass_text_classification_20_newsgroups) shows how to use SageMaker hyperparameter tuning with the pre-built HuggingFace container and 20_newsgroups dataset.
- [Keras BYO Tuning](hyperparameter_tuning/keras_bring_your_own) shows how to use SageMaker hyperparameter tuning with a custom container running a Keras convolutional network on CIFAR-10 data.
- [R BYO Tuning](hyperparameter_tuning/r_bring_your_own) shows how to use SageMaker hyperparameter tuning with the custom container from the [Bring Your Own R Algorithm](advanced_functionality/r_bring_your_own) example.
- [Analyzing Results](hyperparameter_tuning/analyze_results) is a shared notebook that can be used after each of the above notebooks to provide analysis on how training jobs with different hyperparameters performed.

### SageMaker Autopilot

These examples introduce SageMaker Autopilot. Autopilot automatically performs feature engineering, model selection, model tuning (hyperparameter optimization) and allows you to directly deploy the best model to an endpoint to serve inference requests.

- [Customer Churn AutoML](autopilot/) shows how to use SageMaker Autopilot to automatically train a model for the [Predicting Customer Churn](introduction_to_applying_machine_learning/xgboost_customer_churn) task.
- [Targeted Direct Marketing AutoML](autopilot/) shows how to use SageMaker Autopilot to automatically train a model.
- [Housing Prices AutoML](sagemaker-autopilot/housing_prices) shows how to use SageMaker Autopilot for a linear regression problem (predict housing prices).

### Introduction to Amazon Algorithms

These examples provide quick walkthroughs to get you up and running with Amazon SageMaker's custom developed algorithms. Most of these algorithms can train on distributed hardware, scale incredibly well, and are faster and cheaper than popular alternatives.

- [k-means](sagemaker-python-sdk/1P_kmeans_highlevel) is our introductory example for Amazon SageMaker. It walks through the process of clustering MNIST images of handwritten digits using Amazon SageMaker k-means.
- [Factorization Machines](introduction_to_amazon_algorithms/factorization_machines_mnist) showcases Amazon SageMaker's implementation of the algorithm to predict whether a handwritten digit from the MNIST dataset is a 0 or not using a binary classifier.
- [Latent Dirichlet Allocation (LDA)](introduction_to_amazon_algorithms/lda_topic_modeling) introduces topic modeling using Amazon SageMaker Latent Dirichlet Allocation (LDA) on a synthetic dataset.
- [Linear Learner](introduction_to_amazon_algorithms/linear_learner_mnist) predicts whether a handwritten digit from the MNIST dataset is a 0 or not using a binary classifier from Amazon SageMaker Linear Learner.
- [Neural Topic Model (NTM)](introduction_to_amazon_algorithms/ntm_synthetic) uses Amazon SageMaker Neural Topic Model (NTM) to uncover topics in documents from a synthetic data source, where topic distributions are known.
- [Principal Components Analysis (PCA)](introduction_to_amazon_algorithms/pca_mnist) uses Amazon SageMaker PCA to calculate eigendigits from MNIST.
- [Seq2Seq](introduction_to_amazon_algorithms/seq2seq_translation_en-de) uses the Amazon SageMaker Seq2Seq algorithm that's built on top of [Sockeye](https://github.com/awslabs/sockeye), which is a sequence-to-sequence framework for Neural Machine Translation based on MXNet. Seq2Seq implements state-of-the-art encoder-decoder architectures which can also be used for tasks like Abstractive Summarization in addition to Machine Translation. This notebook shows translation from English to German text.
- [Image Classification](introduction_to_amazon_algorithms/imageclassification_caltech) includes full training and transfer learning examples of Amazon SageMaker's Image Classification algorithm. This uses a ResNet deep convolutional neural network to classify images from the caltech dataset.
- [XGBoost for regression](introduction_to_amazon_algorithms/xgboost_abalone) predicts the age of abalone ([Abalone dataset](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression.html)) using regression from Amazon SageMaker's implementation of [XGBoost](https://github.com/dmlc/xgboost).
- [XGBoost for multi-class classification](introduction_to_amazon_algorithms/xgboost_mnist) uses Amazon SageMaker's implementation of [XGBoost](https://github.com/dmlc/xgboost) to classify handwritten digits from the MNIST dataset as one of the ten digits using a multi-class classifier. Both single machine and distributed use-cases are presented.
- [DeepAR for time series forecasting](introduction_to_amazon_algorithms/deepar_synthetic) illustrates how to use the Amazon SageMaker DeepAR algorithm for time series forecasting on a synthetically generated data set.
- [BlazingText Word2Vec](introduction_to_amazon_algorithms/blazingtext_word2vec_text8) generates Word2Vec embeddings from a cleaned text dump of Wikipedia articles using SageMaker's fast and scalable BlazingText implementation.
- [Object detection for bird images](introduction_to_amazon_algorithms/object_detection_birds) demonstrates how to use the Amazon SageMaker Object Detection algorithm with a public dataset of Bird images.
- [Object2Vec for movie recommendation](introduction_to_amazon_algorithms/object2vec_movie_recommendation) demonstrates how Object2Vec can be used to model data consisting of pairs of singleton tokens using movie recommendation as a running example.
- [Object2Vec for multi-label classification](introduction_to_amazon_algorithms/object2vec_multilabel_genre_classification) shows how ObjectToVec algorithm can train on data consisting of pairs of sequences and singleton tokens using the setting of genre prediction of movies based on their plot descriptions.
- [Object2Vec for sentence similarity](introduction_to_amazon_algorithms/object2vec_sentence_similarity) explains how to train Object2Vec using sequence pairs as input using sentence similarity analysis as the application.
- [IP Insights for suspicious logins](introduction_to_amazon_algorithms/ipinsights_login) shows how to train IP Insights on a login events for a web server to identify suspicious login attempts.
- [Semantic Segmentation](introduction_to_amazon_algorithms/semantic_segmentation_pascalvoc) shows how to train a semantic segmentation algorithm using the Amazon SageMaker Semantic Segmentation algorithm. It also demonstrates how to host the model and produce segmentation masks and probability of segmentation.
- [JumpStart Instance Segmentation](introduction_to_amazon_algorithms/jumpstart_instance_segmentation) demonstrates how to use a pre-trained Instance Segmentation model available in JumpStart for inference.
- [JumpStart Semantic Segmentation](introduction_to_amazon_algorithms/jumpstart_semantic_segmentation) demonstrates how to use a pre-trained Semantic Segmentation model available in JumpStart for inference, how to finetune the pre-trained model on a custom dataset using JumpStart transfer learning algorithm, and how to use fine-tuned model for inference.
- [JumpStart Text Generation](introduction_to_amazon_algorithms/jumpstart_text_generation) shows how to use JumpStart to generate text that appears indistinguishable from the hand-written text.
- [JumpStart Text Summarization](introduction_to_amazon_algorithms/jumpstart_text_summarization) shows how to use JumpStart to summarize the text to contain only the important information.
- [JumpStart Image Embedding](introduction_to_amazon_algorithms/jumpstart_image_embedding) demonstrates how to use a pre-trained model available in JumpStart for image embedding.
- [JumpStart Text Embedding](introduction_to_amazon_algorithms/jumpstart_text_embedding) demonstrates how to use a pre-trained model available in JumpStart for text embedding.
- [JumpStart Object Detection](introduction_to_amazon_algorithms/jumpstart_object_detection) demonstrates how to use a pre-trained Object Detection model available in JumpStart for inference, how to finetune the pre-trained model on a custom dataset using JumpStart transfer learning algorithm, and how to use fine-tuned model for inference.
- [JumpStart Machine Translation](introduction_to_amazon_algorithms/jumpstart_machine_translation) demonstrates how to translate text from one language to another language in JumpStart.
- [JumpStart Named Entity Recognition](introduction_to_amazon_algorithms/jumpstart_named_entity_recognition) demonstrates how to identify named entities such as names, locations etc. in the text in JumpStart.

### Amazon SageMaker RL

The following provide examples demonstrating different capabilities of Amazon SageMaker RL.

- [Cartpole using Coach](reinforcement_learning/rl_cartpole_coach) demonstrates the simplest usecase of Amazon SageMaker RL using Intel's RL Coach.
- [AWS DeepRacer](reinforcement_learning/rl_deepracer_robomaker_coach_gazebo) demonstrates AWS DeepRacer trainig using RL Coach in the Gazebo environment.
- [HVAC using EnergyPlus](reinforcement_learning/rl_hvac_coach_energyplus) demonstrates the training of HVAC systems using the EnergyPlus environment.
- [Knapsack Problem](reinforcement_learning/rl_knapsack_coach_custom) demonstrates how to solve the knapsack problem using a custom environment.
- [Mountain Car](reinforcement_learning/rl_mountain_car_coach_gymEnv) Mountain car is a classic RL problem. This notebook explains how to solve this using the OpenAI Gym environment.
- [Distributed Neural Network Compression](reinforcement_learning/rl_network_compression_ray_custom) This notebook explains how to compress ResNets using RL, using a custom environment and the RLLib toolkit.
- [Portfolio Management](reinforcement_learning/rl_portfolio_management_coach_customEnv) This notebook uses a custom Gym environment to manage multiple financial investments.
- [Autoscaling](reinforcement_learning/rl_predictive_autoscaling_coach_customEnv) demonstrates how to adjust load depending on demand. This uses RL Coach and a custom environment.
- [Roboschool](reinforcement_learning/rl_roboschool_ray) is an open source physics simulator that is commonly used to train RL policies for robotic systems. This notebook demonstrates training a few agents using it.
- [Stable Baselines](reinforcement_learning/rl_roboschool_stable_baselines) In this notebook example, we will make the HalfCheetah agent learn to walk using the stable-baselines, which are a set of improved implementations of Reinforcement Learning (RL) algorithms based on OpenAI Baselines.
- [Travelling Salesman](reinforcement_learning/rl_traveling_salesman_vehicle_routing_coach) is a classic NP hard problem, which this notebook solves with AWS SageMaker RL.
- [Tic-tac-toe](reinforcement_learning/rl_tic_tac_toe_coach_customEnv) is a simple implementation of a custom Gym environment to train and deploy an RL agent in Coach that then plays tic-tac-toe interactively in a Jupyter Notebook.
- [Unity Game Agent](reinforcement_learning/rl_unity_ray) shows how to use RL algorithms to train an agent to play Unity3D game.

### Scientific Details of Algorithms

These examples provide more thorough mathematical treatment on a select group of algorithms.

- [Streaming Median](scientific_details_of_algorithms/streaming_median) sequentially introduces concepts used in streaming algorithms, which many SageMaker algorithms rely on to deliver speed and scalability.
- [Latent Dirichlet Allocation (LDA)](scientific_details_of_algorithms/lda_topic_modeling) dives into Amazon SageMaker's spectral decomposition approach to LDA.
- [Linear Learner features](scientific_details_of_algorithms/linear_learner_class_weights_loss_functions) shows how to use the class weights and loss functions features of the SageMaker Linear Learner algorithm to improve performance on a credit card fraud prediction task

### Amazon SageMaker Debugger

These examples provide and introduction to SageMaker Debugger which allows debugging and monitoring capabilities for training of machine learning and deep learning algorithms. Note that although these notebooks focus on a specific framework, the same approach works with all the frameworks that Amazon SageMaker Debugger supports. The notebooks below are listed in the order in which we recommend you review them.

- [Using a built-in rule with TensorFlow](sagemaker-debugger/tensorflow_builtin_rule/)
- [Using a custom rule with TensorFlow Keras](sagemaker-debugger/tensorflow_keras_custom_rule/)
- [Interactive tensor analysis in notebook with MXNet](sagemaker-debugger/mnist_tensor_analysis/)
- [Visualizing Debugging Tensors of MXNet training](sagemaker-debugger/mnist_tensor_plot/)
- [Real-time analysis in notebook with MXNet](sagemaker-debugger/mxnet_realtime_analysis/)
- [Using a built in rule with XGBoost](sagemaker-debugger/xgboost_builtin_rules/)
- [Real-time analysis in notebook with XGBoost](sagemaker-debugger/xgboost_realtime_analysis/)
- [Using SageMaker Debugger with Managed Spot Training and MXNet](sagemaker-debugger/mxnet_spot_training/)
- [Reacting to CloudWatch Events from Rules to take an action based on status with TensorFlow](sagemaker-debugger/tensorflow_action_on_rule/)
- [Using SageMaker Debugger with a custom PyTorch container](sagemaker-debugger/pytorch_custom_container/)

### Amazon SageMaker Clarify

These examples provide an introduction to SageMaker Clarify which provides machine learning developers with greater visibility into their training data and models so they can identify and limit bias and explain predictions.

* [Fairness and Explainability with SageMaker Clarify](sagemaker_processing/fairness_and_explainability) shows how to use SageMaker Clarify Processor API to measure the pre-training bias of a dataset and post-training bias of a model, and explain the importance of the input features on the model's decision.
* [Amazon SageMaker Clarify Model Monitors](sagemaker_model_monitor/fairness_and_explainability) shows how to use SageMaker Clarify Model Monitor API to schedule bias monitor to monitor predictions for bias drift on a regular basis, and schedule explainability monitor to monitor predictions for feature attribution drift on a regular basis.

### Publishing content from RStudio on Amazon SageMaker to RStudio Connect

These examples show you how to run R examples, and publish applications in RStudio on Amazon SageMaker to RStudio Connect.

- [Publishing R Markdown](r_examples/rsconnect_rmarkdown/) shows how you can author an R Markdown document (.Rmd, .Rpres) within RStudio on Amazon SageMaker and publish to RStudio Connect for wide consumption.
- [Publishing R Shiny Apps](r_examples/rsconnect_shiny/) shows how you can author an R Shiny application within RStudio on Amazon SageMaker and publish to RStudio Connect for wide consumption.
- [Publishing Streamlit Apps](r_examples/rsconnect_streamlit/) shows how you can author a streamlit application withing Amazon SageMaker Studio and publish to RStudio Connect for wide consumption.

### Advanced Amazon SageMaker Functionality

These examples showcase unique functionality available in Amazon SageMaker. They cover a broad range of topics and utilize a variety of methods, but aim to provide the user with sufficient insight or inspiration to develop within Amazon SageMaker.

- [Data Distribution Types](advanced_functionality/data_distribution_types) showcases the difference between two methods for sending data from S3 to Amazon SageMaker Training instances. This has particular implication for scalability and accuracy of distributed training.
- [Encrypting Your Data](advanced_functionality/handling_kms_encrypted_data) shows how to use Server Side KMS encrypted data with Amazon SageMaker training. The IAM role used for S3 access needs to have permissions to encrypt and decrypt data with the KMS key.
- [Using Parquet Data](advanced_functionality/parquet_to_recordio_protobuf) shows how to bring [Parquet](https://parquet.apache.org/) data sitting in S3 into an Amazon SageMaker Notebook and convert it into the recordIO-protobuf format that many SageMaker algorithms consume.
- [Connecting to Redshift](advanced_functionality/working_with_redshift_data) demonstrates how to copy data from Redshift to S3 and vice-versa without leaving Amazon SageMaker Notebooks.
- [Bring Your Own XGBoost Model](advanced_functionality/xgboost_bring_your_own_model) shows how to use Amazon SageMaker Algorithms containers to bring a pre-trained model to a realtime hosted endpoint without ever needing to think about REST APIs.
- [Bring Your Own k-means Model](advanced_functionality/kmeans_bring_your_own_model) shows how to take a model that's been fit elsewhere and use Amazon SageMaker Algorithms containers to host it.
- [Bring Your Own R Algorithm](advanced_functionality/r_bring_your_own) shows how to bring your own algorithm container to Amazon SageMaker using the R language.
- [Installing the R Kernel](advanced_functionality/install_r_kernel) shows how to install the R kernel into an Amazon SageMaker Notebook Instance.
- [Bring Your Own scikit Algorithm](advanced_functionality/scikit_bring_your_own) provides a detailed walkthrough on how to package a scikit learn algorithm for training and production-ready hosting.
- [Bring Your Own MXNet Model](advanced_functionality/mxnet_mnist_byom) shows how to bring a model trained anywhere using MXNet into Amazon SageMaker.
- [Bring Your Own TensorFlow Model](advanced_functionality/tensorflow_iris_byom) shows how to bring a model trained anywhere using TensorFlow into Amazon SageMaker.
- [Experiment Management Capabilities with Search](advanced_functionality/search) shows how to organize Training Jobs into projects, and track relationships between Models, Endpoints, and Training Jobs.
- [Host Multiple Models with Your Own Algorithm](advanced_functionality/multi_model_bring_your_own) shows how to deploy multiple models to a realtime hosted endpoint with your own custom algorithm.
- [Host Multiple Models with XGBoost](advanced_functionality/multi_model_xgboost_home_value) shows how to deploy multiple models to a realtime hosted endpoint using a multi-model enabled XGBoost container.
- [Host Multiple Models with SKLearn](advanced_functionality/multi_model_sklearn_home_value) shows how to deploy multiple models to a realtime hosted endpoint using a multi-model enabled SKLearn container.
- [SageMaker Training and Inference with Script Mode](sagemaker-script-mode) shows how to use custom training and inference scripts, similar to those you would use outside of SageMaker, with SageMaker's prebuilt containers for various frameworks like Scikit-learn, PyTorch, and XGBoost.
- [Host Models with NVidia Triton Server](sagemaker-triton) shows how to deploy models to a realtime hosted endpoint using [Triton](https://developer.nvidia.com/nvidia-triton-inference-server) as the model inference server.

### Amazon SageMaker Neo Compilation Jobs

These examples provide an introduction to how to use Neo to compile and optimize deep learning models.

- [GluonCV SSD Mobilenet](sagemaker_neo_compilation_jobs/gluoncv_ssd_mobilenet) shows how to train GluonCV SSD MobileNet and use Amazon SageMaker Neo to compile and optimize the trained model.
- [Image Classification](sagemaker_neo_compilation_jobs/imageclassification_caltech) Adapts from [image classification](introduction_to_amazon_algorithms/imageclassification_caltech) including Neo API and comparison against the uncompiled baseline.
- [MNIST with MXNet](sagemaker_neo_compilation_jobs/mxnet_mnist) Adapts from [MXNet MNIST](sagemaker-python-sdk/mxnet_mnist) including Neo API and comparison against the uncompiled baseline.
- [Deploying pre-trained PyTorch vision models](sagemaker_neo_compilation_jobs/pytorch_torchvision) shows how to use Amazon SageMaker Neo to compile and optimize pre-trained PyTorch models from TorchVision.
- [Distributed TensorFlow](sagemaker_neo_compilation_jobs/tensorflow_distributed_mnist) includes Neo API and comparison against the uncompiled baseline.
- [Predicting Customer Churn](sagemaker_neo_compilation_jobs/xgboost_customer_churn) Adapts from [XGBoost customer churn](introduction_to_applying_machine_learning/xgboost_customer_churn) including Neo API and comparison against the uncompiled baseline.

### Amazon SageMaker Processing

These examples show you how to use SageMaker Processing jobs to run data processing workloads.

- [Scikit-Learn Data Processing and Model Evaluation](sagemaker_processing/scikit_learn_data_processing_and_model_evaluation) shows how to use SageMaker Processing and the Scikit-Learn container to run data preprocessing and model evaluation workloads.
- [Feature transformation with Amazon SageMaker Processing and SparkML](sagemaker_processing/feature_transformation_with_sagemaker_processing) shows how to use SageMaker Processing to run data processing workloads using SparkML prior to training.
- [Feature transformation with Amazon SageMaker Processing and Dask](sagemaker_processing/feature_transformation_with_sagemaker_processing_dask) shows how to use SageMaker Processing to transform data using Dask distributed clusters
- [Distributed Data Processing using Apache Spark and SageMaker Processing](sagemaker_processing/spark_distributed_data_processing) shows how to use the built-in Spark container on SageMaker Processing using the SageMaker Python SDK.

### Amazon SageMaker Pipelines

These examples show you how to use [SageMaker Pipelines](https://aws.amazon.com/sagemaker/pipelines) to create, automate and manage end-to-end Machine Learning workflows.

- [Amazon Comprehend with SageMaker Pipelines](sagemaker-pipelines/nlp/amazon_comprehend_sagemaker_pipeline) shows how to deploy a custom text classification using Amazon Comprehend and SageMaker Pipelines.
- [Amazon Forecast with SageMaker Pipelines](sagemaker-pipelines/time_series_forecasting/amazon_forecast_pipeline) shows how you can create a dataset, dataset group and predictor with Amazon Forecast and SageMaker Pipelines.
- [Multi-model SageMaker Pipeline with Hyperparamater Tuning and Experiments](sagemaker-pipeline-multi-model) shows how you can generate a regression model by training real estate data from Athena using Data Wrangler, and uses multiple algorithms both from a custom container and a SageMaker container in a single pipeline.

### Amazon SageMaker Pre-Built Framework Containers and the Python SDK

#### Pre-Built Deep Learning Framework Containers

These examples show you how to train and host in pre-built deep learning framework containers using the SageMaker Python SDK.

- [Chainer CIFAR-10](sagemaker-python-sdk/chainer_cifar10) trains a VGG image classification network on CIFAR-10 using Chainer (both single machine and multi-machine versions are included)
- [Chainer MNIST](sagemaker-python-sdk/chainer_mnist) trains a basic neural network on MNIST using Chainer (shows how to use local mode)
- [Chainer sentiment analysis](sagemaker-python-sdk/chainer_sentiment_analysis) trains a LSTM network with embeddings to predict text sentiment using Chainer
- [IRIS with Scikit-learn](sagemaker-python-sdk/scikit_learn_iris) trains a Scikit-learn classifier on IRIS data
- [Model Registry and Batch Transform with Scikit-learn](sagemaker-python-sdk/scikit_learn_model_registry_batch_transform) trains a Scikit-learn Random Forest model, registers it in Model Registry, and runs a Batch Transform Job.
- [MNIST with MXNet Gluon](sagemaker-python-sdk/mxnet_gluon_mnist) trains a basic neural network on the MNIST handwritten digit dataset using MXNet Gluon
- [MNIST with MXNet](sagemaker-python-sdk/mxnet_mnist) trains a basic neural network on the MNIST handwritten digit data using MXNet's symbolic syntax
- [Sentiment Analysis with MXNet Gluon](sagemaker-python-sdk/mxnet_gluon_sentiment) trains a text classifier using embeddings with MXNet Gluon
- [TensorFlow training and serving](sagemaker-python-sdk/tensorflow_script_mode_training_and_serving) trains a basic neural network on MNIST
- [TensorFlow with Horovod](sagemaker-python-sdk/tensorflow_script_mode_horovod) trains on MNIST using Horovod for distributed training
- [TensorFlow using shell commands](sagemaker-python-sdk/tensorflow_script_mode_using_shell_commands) shows how to use a shell script for the container's entry point

#### Pre-Built Machine Learning Framework Containers

These examples show you how to build Machine Learning models with frameworks like Apache Spark or Scikit-learn using SageMaker Python SDK.

- [Inference with SparkML Serving](sagemaker-python-sdk/sparkml_serving_emr_mleap_abalone) shows how to build an ML model with Apache Spark using Amazon EMR on Abalone dataset and deploy in SageMaker with SageMaker SparkML Serving.
- [Pipeline Inference with Scikit-learn and LinearLearner](sagemaker-python-sdk/scikit_learn_inference_pipeline) builds a ML pipeline using Scikit-learn preprocessing and LinearLearner algorithm in single endpoint

### Using Amazon SageMaker with Apache Spark

These examples show how to use Amazon SageMaker for model training, hosting, and inference through Apache Spark using [SageMaker Spark](https://github.com/aws/sagemaker-spark). SageMaker Spark allows you to interleave Spark Pipeline stages with Pipeline stages that interact with Amazon SageMaker.

- [MNIST with SageMaker PySpark](sagemaker-spark/pyspark_mnist)

### Using Amazon SageMaker with Amazon Keyspaces (for Apache Cassandra)

These examples show how to use Amazon SageMaker to read data from [Amazon Keyspaces](https://docs.aws.amazon.com/keyspaces/).
- [Train Machine Learning Models using Amazon Keyspaces as a Data Source](ingest_data/sagemaker-keyspaces)


### AWS Marketplace

#### Create algorithms/model packages for listing in AWS Marketplace for machine learning.

These example notebooks show you how to package a model or algorithm for listing in AWS Marketplace for machine learning.

- [Creating Marketplace Products](aws_marketplace/creating_marketplace_products)
  - [Creating a Model Package - Listing on AWS Marketplace](aws_marketplace/creating_marketplace_products/models) provides a detailed walkthrough on how to package a pre-trained model as a SageMaker Model Package that can be listed on AWS Marketplace.
  - [Creating Algorithm and Model Package - Listing on AWS Marketplace](aws_marketplace/creating_marketplace_products/algorithms) provides a detailed walkthrough on how to package a scikit learn algorithm to create SageMaker Algorithm and SageMaker Model Package entities that can be used with the enhanced SageMaker Train/Transform/Hosting/Tuning APIs and listed on AWS Marketplace.

Once you have created an algorithm or a model package to be listed in the AWS Marketplace, the next step is to list it in AWS Marketplace, and provide a sample notebook that customers can use to try your algorithm or model package.

- [Curate your AWS Marketplace model package listing and sample notebook](aws_marketplace/curating_aws_marketplace_listing_and_sample_notebook/ModelPackage) provides instructions on how to craft a sample notebook to be associated with your listing and how to curate a good AWS Marketplace listing that makes it easy for AWS customers to consume your model package.
- [Curate your AWS Marketplace algorithm listing and sample notebook](aws_marketplace/curating_aws_marketplace_listing_and_sample_notebook/Algorithm) provides instructions on how to craft a sample notebook to be associated with your listing and how to curate a good AWS Marketplace listing that makes it easy for your customers to consume your algorithm.

#### Use algorithms, data, and model packages from AWS Marketplace.

These examples show you how to use model-packages and algorithms from AWS Marketplace and dataset products from AWS Data Exchange, for machine learning.

- [Using Algorithms](aws_marketplace/using_algorithms)
  - [Using Algorithm From AWS Marketplace](aws_marketplace/using_algorithms/amazon_demo_product) provides a detailed walkthrough on how to use Algorithm with the enhanced SageMaker Train/Transform/Hosting/Tuning APIs by choosing a canonical product listed on AWS Marketplace.
  - [Using AutoML algorithm](aws_marketplace/using_algorithms/automl) provides a detailed walkthrough on how to use AutoML algorithm from AWS Marketplace.
- [Using Model Packages](aws_marketplace/using_model_packages)
  - [Using Model Packages From AWS Marketplace](aws_marketplace/using_model_packages/generic_sample_notebook) is a generic notebook which provides sample code snippets you can modify and use for performing inference on Model Packages from AWS Marketplace, using Amazon SageMaker.
  - [Using Amazon Demo product From AWS Marketplace](aws_marketplace/using_model_packages/amazon_demo_product) provides a detailed walkthrough on how to use Model Package entities with the enhanced SageMaker Transform/Hosting APIs by choosing a canonical product listed on AWS Marketplace.
  - [Using models for extracting vehicle metadata](aws_marketplace/using_model_packages/auto_insurance) provides a detailed walkthrough on how to use pre-trained models from AWS Marketplace for extracting metadata for a sample use-case of auto-insurance claim processing.
  - [Using models for identifying non-compliance at a workplace](aws_marketplace/using_model_packages/improving_industrial_workplace_safety) provides a detailed walkthrough on how to use pre-trained models from AWS Marketplace for extracting metadata for a sample use-case of generating summary reports for identifying non-compliance at a construction/industrial workplace.
  - [Creative writing using GPT-2 Text Generation](aws_marketplace/using_model_packages/creative-writing-using-gpt-2-text-generation) will show you how to use AWS Marketplace GPT-2-XL pre-trained model on Amazon SageMaker to generate text based on your prompt to help you author prose and poetry.
  - [Amazon Augmented AI with AWS Marketplace ML models](aws_marketplace/using_model_packages/amazon_augmented_ai_with_aws_marketplace_ml_models) will show you how to use AWS Marketplace pre-trained ML models with Amazon Augmented AI to implement human-in-loop workflow reviews with your ML model predictions.
  - [Monitoring data quality in third-party models from AWS Marketplace](aws_marketplace/using_model_packages/data_quality_monitoring) will show you how to perform Data Quality monitoring on a pre-trained third-party model from AWS Marketplace.
  - [Evaluating ML models from AWS Marketplace for person counting use case](aws_marketplace/using_model_packages/evaluating_aws_marketplace_models_for_person_counting_use_case) will show you how to use two AWS Marketplace GluonCV pre-trained ML models for person counting use case and evaluate each model for performance in different types of crowd images.
  - [Preprocessing audio data using a pre-trained machine learning model](using_model_packages/preprocessing-audio-data-using-a-machine-learning-model) demonstrates the usage of a pre-trained audio track separation model to create synthetic features and improve an acoustic classification model.
- [Using Dataset Products](aws_marketplace/using_data)
  - [Using Dataset Product from AWS Data Exchange with ML model from AWS Marketplace](aws_marketplace/using_data/using_data_with_ml_model) is a sample notebook which shows how a dataset from AWS Data Exchange can be used with an ML Model Package from AWS Marketplace.
  - [Using Shutterstock Image Datasets to train Image Classification Models](aws_marketplace/using_data/image_classification_with_shutterstock_image_datasets) provides a detailed walkthrough on how to use the [Free Sample: Images & Metadata of “Whole Foods” Shoppers](https://aws.amazon.com/marketplace/pp/prodview-y6xuddt42fmbu?qid=1623195111604&sr=0-1&ref_=srh_res_product_title#offers) from Shutterstock's Image Datasets to train a multi-label image classification model using Shutterstock's pre-labeled image assets. You can learn more about this implementation [from this blog post](https://aws.amazon.com/blogs/awsmarketplace/using-shutterstocks-image-datasets-to-train-your-computer-vision-models/).

## :balance_scale: License

This library is licensed under the [Apache 2.0 License](http://aws.amazon.com/apache2.0/).
For more details, please take a look at the [LICENSE](https://github.com/aws/amazon-sagemaker-examples/blob/master/LICENSE.txt) file.

## :handshake: Contributing

Although we're extremely excited to receive contributions from the community, we're still working on the best mechanism to take in examples from external sources. Please bear with us in the short-term if pull requests take longer than expected or are closed.
Please read our [contributing guidelines](https://github.com/aws/amazon-sagemaker-examples/blob/master/CONTRIBUTING.md)
if you'd like to open an issue or submit a pull request.
