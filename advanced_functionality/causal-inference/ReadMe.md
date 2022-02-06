# Building your own algorithm container for Causal Inference

## The Structure of the Sample Code

The components are as follows:
* __causal_inference_container.ipynb__: The notebook example on how to build container for causal inference using [CausalNex][causal-nes]

* __container__/__Dockerfile__: The _Dockerfile_ describes how the image is built and what it contains. It is a recipe for your container and gives you tremendous flexibility to construct almost any execution environment you can imagine. Here. we use the Dockerfile to describe a pretty standard python science stack and the simple scripts that we're going to add to it. See the [Dockerfile reference][dockerfile] for what's possible here.

* __container__/__build\_and\_push.sh__: The script to build the Docker image (using the Dockerfile above) and push it to the [Amazon EC2 Container Registry (ECR)][ecr] so that it can be deployed to SageMaker. Specify the name of the image as the argument to this script. The script will generate a full name for the repository in your account and your configured AWS region. If this ECR repository doesn't exist, the script will create it.

* __container__/__causal_nex__: The directory that contains the application to run in the container. See the next session for details about each of the files.

* __container__/__local-test__: A directory containing scripts and a setup for running a simple training and inference jobs locally so that you can test that everything is set up correctly. See below for details.

### The application run inside the container

When SageMaker starts a container, it will invoke the container with an argument of either __train__ or __serve__. We have set this container up so that the argument in treated as the command that the container executes. When training, it will run the __train__ program included and, when serving, it will run the __serve__ program.

* __train__: The main program for training the model. When you build your own algorithm, you'll edit this to include your training code.
* __serve__: The wrapper that starts the inference server. In most cases, you can use this file as-is.
* __wsgi.py__: The start up shell for the individual server workers. This only needs to be changed if you changed where predictor.py is located or is named.
* __predictor.py__: The algorithm-specific inference server. This is the file that you modify with your own algorithm's code.
* __nginx.conf__: The configuration for the nginx master server that manages the multiple workers.

### Setup for local testing

The subdirectory local-test contains scripts and sample data for testing the built container image on the local machine. When building your own algorithm, you'll want to modify it appropriately.

* __train-local.sh__: Instantiate the container configured for training.
* __serve-local.sh__: Instantiate the container configured for serving.
* __predict.sh__: Run predictions against a locally instantiated server.
* __test-dir__: The directory that gets mounted into the container with test data mounted in all the places that match the container schema.
* __payload.json__: Sample data for used by predict.sh for testing the server.

#### The directory tree mounted into the container

The tree under test-dir is mounted into the container and mimics the directory structure that SageMaker would create for the running container during training or hosting.

* __input/data/training/heart_failure_clinical_records_dataset.csv__: The training data.
* __model__: The directory where the algorithm writes the model file.
* __output__: The directory where the algorithm can write its success or failure file.


#### Local testing of the container
To locally debug the container, without creating the endpoint, open one terminal and run:
bash container/local_test/serve_local.sh causal-nex-container

With open terminal, open second terminal and run:
bash container/local_test/predict.sh *your_test_file.json*

## Environment variables

When you create an inference server, you can control some of Gunicorn's options via environment variables. These
can be supplied as part of the CreateModel API call.

    Parameter                Environment Variable              Default Value
    ---------                --------------------              -------------
    number of workers        MODEL_SERVER_WORKERS              the number of CPU cores
    timeout                  MODEL_SERVER_TIMEOUT              60 seconds

## Dataset
Davide Chicco, Giuseppe Jurman: “Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone”. BMC Medical Informatics and Decision Making 20, 16 (2020). 
[Web Link]: https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-020-1023-5 

[skl]: http://scikit-learn.org "scikit-learn Home Page"
[dockerfile]: https://docs.docker.com/engine/reference/builder/ "The official Dockerfile reference guide"
[ecr]: https://aws.amazon.com/ecr/ "ECR Home Page"
[nginx]: http://nginx.org/
[gunicorn]: http://gunicorn.org/
[flask]: http://flask.pocoo.org/
[causal-nes]: https://causalnex.readthedocs.io/en/latest/

