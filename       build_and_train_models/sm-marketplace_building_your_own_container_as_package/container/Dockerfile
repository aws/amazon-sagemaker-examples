# Build an image that can do training and inference in SageMaker.
# This uses a Python 3 image with the nginx, gunicorn, flask stack
# for serving inferences in a stable way.

FROM python:3.10-slim-bullseye

LABEL maintainer="Amazon AI <sage-learner@amazon.com>"

# nginx + ca-certificates are needed for the serving stack. Python/pip come with the base image.
RUN apt-get -y update && apt-get install -y --no-install-recommends \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install the Python packages needed by scikit-learn training and the flask serving stack.
# pip leaves the install caches populated which uses a significant amount of space, so we
# disable the cache to keep the image small and fast to start.
RUN pip install --no-cache-dir numpy scipy scikit-learn pandas flask gevent gunicorn

# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
COPY decision_trees /opt/program
WORKDIR /opt/program
