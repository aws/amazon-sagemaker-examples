
From 763104351884.dkr.ecr.us-east-2.amazonaws.com/mxnet-training:1.6.0-gpu-py36-cu101-ubuntu16.04

RUN pip install gluonnlp pandas
RUN pip install spacy 
RUN python3 -m spacy download en
