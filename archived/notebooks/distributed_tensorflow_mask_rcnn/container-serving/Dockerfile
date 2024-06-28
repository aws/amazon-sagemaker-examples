FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:2.12.0-gpu-py310-cu118-ubuntu20.04-sagemaker

RUN pip3 install --upgrade pip
RUN pip3 install ujson==5.7.0
RUN pip3 install scikit-image==0.19.3
RUN pip3 install tk==0.1.0

RUN git clone https://github.com/tensorpack/tensorpack.git /tensorpack
RUN cd /tensorpack && git fetch origin fac024f0f72fd593ea243f0b599a51b11fe4effd 
RUN cd /tensorpack && git reset --hard fac024f0f72fd593ea243f0b599a51b11fe4effd 
RUN pip3 install -e /tensorpack 

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
ADD  data /
ENV WORKDIR /

ENTRYPOINT ["python", "/serve.py"]
