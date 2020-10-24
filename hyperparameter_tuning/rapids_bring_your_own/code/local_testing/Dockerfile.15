FROM rapidsai/rapidsai:0.15-cuda11.0-base-ubuntu18.04-py3.7 

ENV DATASET_DIRECTORY="1_year"
ENV ALGORITHM_CHOICE="XGBoost"
ENV ML_WORKFLOW_CHOICE="singleGPU"
ENV CV_FOLDS="3"

# ensure printed output/log-messages retain correct order
ENV PYTHONUNBUFFERED=True

# add sagemaker-training-toolkit [ requires build tools ], flask [ serving ], and dask-ml
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \ 
    && source activate rapids && pip3 install sagemaker-training \
    && conda install -c anaconda flask \
    && conda install -c conda-forge dask-ml

# path where SageMaker looks for code when container runs in the cloud
ENV CLOUD_PATH="/opt/ml/code"

# copy our latest [local] code into the container 
COPY . $CLOUD_PATH

# make the entrypoint script executable
RUN chmod +x $CLOUD_PATH/entrypoint.sh

WORKDIR $CLOUD_PATH
ENTRYPOINT ["./entrypoint.sh"]
