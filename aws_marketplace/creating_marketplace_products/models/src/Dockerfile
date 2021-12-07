FROM ubuntu:18.04

# Specify encoding
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Install python-pip
RUN apt-get update \
&& apt-get install -y python3.6 python3-pip curl \
&& ln -s /usr/bin/python3.6 /usr/bin/python \
&& ln -s /usr/bin/pip3 /usr/bin/pip;

# Install flask server
RUN pip install -U flask gunicorn joblib sklearn;

# Copy scoring logic and model artifacts into the docker image
COPY scoring_logic.py /scoring_logic.py
COPY wsgi.py /wsgi.py
COPY model-artifacts.joblib /opt/ml/model/model-artifacts.joblib
COPY serve /opt/program/serve

RUN chmod 755 /opt/program/serve
ENV PATH=/opt/program:${PATH}
