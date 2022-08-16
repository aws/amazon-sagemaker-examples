FROM rapidsai/rapidsai:21.06-cuda11.0-base-ubuntu18.04-py3.7 

ENV DATASET_DIRECTORY="3_year"
ENV ALGORITHM_CHOICE="XGBoost"
ENV ML_WORKFLOW_CHOICE="singleGPU"
ENV CV_FOLDS="3"

# ensure printed output/log-messages retain correct order
ENV PYTHONUNBUFFERED=True

# delete expired nvidia keys and fetch new ones
RUN apt-key del 7fa2af80
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb && dpkg -i cuda-keyring_1.0-1_all.deb 

# add sagemaker-training-toolkit [ requires build tools ], flask [ serving ], and dask-ml
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \ 
    && source activate rapids && pip3 install sagemaker-training dask-ml flask

# path where SageMaker looks for code when container runs in the cloud
ENV CLOUD_PATH "/opt/ml/code"

# copy our latest [local] code into the container 
COPY . $CLOUD_PATH

# make the entrypoint script executable
RUN chmod +x $CLOUD_PATH/entrypoint.sh

WORKDIR $CLOUD_PATH
ENTRYPOINT ["./entrypoint.sh"]
