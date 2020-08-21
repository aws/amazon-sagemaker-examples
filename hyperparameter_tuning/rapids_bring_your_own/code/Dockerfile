FROM rapidsai/rapidsai:0.14-cuda10.1-runtime-ubuntu18.04-py3.7

# install https://github.com/aws/sagemaker-training-toolkit
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \ 
    && source activate rapids && pip3 install sagemaker-training

# path where sagemaker looks for our code
ENV CLOUD_PATH="/opt/ml/code"

# copy our latest [local] code into the container 
COPY rapids_cloud_ml.py $CLOUD_PATH/rapids_cloud_ml.py
COPY train.py $CLOUD_PATH/train.py

# sagemaker entrypoint will be train.py
ENV SAGEMAKER_PROGRAM train.py 

WORKDIR $CLOUD_PATH
