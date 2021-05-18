FROM continuumio/miniconda:latest 

# SageMaker uses /opt/ml for input / output data 
# throughout the training 
RUN mkdir -p /opt/ml

# Copy the training script into /usr/bin 
# as an executable
COPY train.py /usr/bin/train

# make /opt/ml/program/train an executable
RUN chmod +x /usr/bin/train


