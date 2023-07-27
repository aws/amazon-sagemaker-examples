FROM 763104351884.dkr.ecr.eu-west-1.amazonaws.com/pytorch-inference:1.8.1-cpu-py36-ubuntu18.04

RUN pip install --no-cache numpy \
                           transformers==4.5.1  

COPY predict.py /opt/sources/predict.py

ENV PYTHONPATH="$PYTHONPATH:/opt/sources"