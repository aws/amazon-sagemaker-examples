# The base image used is only for demonstration purposes. You could customize this to your usecase
FROM python:3.9-slim-buster

# Set a docker label to enable container to use SAGEMAKER_BIND_TO_PORT environment variable if present
LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true

# Install necessary dependencies along with nginx
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    ca-certificates \
    wget \
    nginx \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to the latest version
RUN pip3 --no-cache-dir install --upgrade pip

# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Copy requirements.txt to /opt/program folder
COPY requirements.txt /opt/program/requirements.txt

# Install packages listed in requirements.txt
RUN pip3 install --no-cache-dir -r /opt/program/requirements.txt

# Copy contents of code/ dir to /opt/program
COPY code/ /opt/program/

# Set working dir to /opt/program which has the serve and inference.py scripts
WORKDIR /opt/program

# Expose port 8080 for serving
EXPOSE 8080

ENTRYPOINT ["python"]

# serve is a python script under code/ directory that launches nginx and gunicorn processes
CMD [ "serve" ]