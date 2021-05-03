FROM continuumio/miniconda3:latest 

ARG ACCESS_KEY
ARG SECRET_KEY
ARG REGION_NAME

# SageMaker uses /opt/ml for input / output data 
# throughout the training 
RUN mkdir -p /opt/ml

# build your aws credentials into 
# the image
ENV ACCESS_KEY=$ACCESS_KEY
ENV SECRET_KEY=$SECRET_KEY
ENV REGION_NAME=$REGION_NAME

# Install dependecies
RUN pip install cryptography
RUN pip install boto3

# Copy the training script into /usr/bin 
# as an executable
COPY train.py /usr/bin/train

# make /opt/ml/program/train an executable
RUN chmod +x /usr/bin/train


