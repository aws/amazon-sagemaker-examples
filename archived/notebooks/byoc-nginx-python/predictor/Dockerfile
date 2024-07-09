FROM ubuntu:18.04

# Set a docker label to advertise multi-model support on the container. Uncomment this if you're using MMS
# LABEL com.amazonaws.sagemaker.capabilities.multi-models=true

# Set a docker label to enable container to use SAGEMAKER_BIND_TO_PORT environment variable if present
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true

# Upgrade installed packages
RUN apt-get update && apt-get upgrade -y && apt-get clean

# Python package management and basic dependencies
RUN apt-get install -y curl python3.7 python3.7-dev python3.7-distutils

# Register the version in alternatives
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1

# Set python 3 as the default python
RUN update-alternatives --set python /usr/bin/python3.7

# Install necessary dependencies and nginx web server
RUN apt-get -y install --no-install-recommends \
    build-essential \
    ca-certificates \
    vim \
    nginx \
    && rm -rf /var/lib/apt/lists/* \
    && python --version \
    && curl -O https://bootstrap.pypa.io/get-pip.py \
    && python get-pip.py

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
RUN update-alternatives --install /usr/local/bin/pip pip /usr/local/bin/pip3 1

# Upgrade pip to the latest version
RUN pip3 --no-cache-dir install --upgrade pip

# install required dependencies including flask, gunicorn, xgboost etc.,
RUN pip3 --no-cache-dir install \
    flask \
    gunicorn \
    gevent \
    numpy \
    pandas \
    joblib \
    xgboost


# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Copy contents of code/ dir to /opt/program
COPY code /opt/program

# Set working dir to /opt/program which has the serve and inference.py scripts
WORKDIR /opt/program

# Expose port 8080 for serving
EXPOSE 8080

ENTRYPOINT ["python"]

# serve is a python script under code/ directory that launches nginx and gunicorn processes
CMD ["serve"]