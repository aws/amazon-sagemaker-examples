FROM --platform=linux/amd64 python:3.10-slim-buster as build

# Copy requirements.txt and install dependencies
COPY requirements.txt /opt/program/requirements.txt
RUN pip3 install -r /opt/program/requirements.txt

# Set working directory and copy application files
WORKDIR /opt/program
COPY src /opt/program

ENV DOCKER_CONTAINER=1 EVAL_RESULTS_PATH=/opt/ml/processing/output/

# Set execute permission for main.py
RUN chmod +x /opt/program/main.py

# Set entrypoint to main.py
ENTRYPOINT ["python3", "/opt/program/main.py"]
