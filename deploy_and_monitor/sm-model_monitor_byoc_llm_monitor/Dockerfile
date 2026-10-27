FROM --platform=linux/amd64 ubuntu:22.04 as build

# Install required packages
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    build-essential \
    libssl-dev \
    libffi-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set the default Python version to 3.10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN update-alternatives --config python3

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