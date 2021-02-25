# build a docker image from conda base
# with directory structure typical to SageMaker containers

# Data is hosted in
# /opt/ml/data

# Training script in 
# /opt/ml/program

# Checkpoints in 
# /opt/ml/checkpoint

FROM 688520471316.dkr.ecr.us-west-2.amazonaws.com/conda:base 

# make directories for input data, script and checkpoint
RUN mkdir -p /opt/ml

# copy the script into /opt/ml/program
# the COPY commands copies everyting in decision_trees/
# into /opt/ml/program but not the directory decision_trees/ itself
# this means /opt/ml/program "looks like" decision_trees 
COPY trivial_training/train.py /usr/bin/train

# make /opt/ml/program/train an executable
RUN chmod +x /usr/bin/train

# set working dir to /opt/ml/program
# WORKDIR /opt/ml/program



