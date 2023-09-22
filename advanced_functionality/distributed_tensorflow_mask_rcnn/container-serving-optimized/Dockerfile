FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:2.12.0-gpu-py310-cu118-ubuntu20.04-sagemaker

RUN pip3 install --upgrade pip
RUN pip3 install ujson==5.7.0
RUN pip3 install scikit-image==0.19.3
RUN pip3 install tk==0.1.0

RUN git clone https://github.com/aws-samples/mask-rcnn-tensorflow.git
RUN cd /mask-rcnn-tensorflow && git fetch origin 59168dc01b022e9039ce66c75178b149922458e2 
RUN cd /mask-rcnn-tensorflow && git reset --hard 59168dc01b022e9039ce66c75178b149922458e2 
RUN chmod -R +w /mask-rcnn-tensorflow
RUN pip3 install -e /mask-rcnn-tensorflow/

##########################################################################################
# SageMaker requirements
##########################################################################################
## install flask
RUN pip install flask

### Install nginx notebook
RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*
 
COPY resources/*.* /
ENV WORKDIR /

ENTRYPOINT ["python", "/serve.py"]
