# Build an image that can do training and inference in SageMaker
# This is a Python 3 image that uses the nginx, gunicorn, flask stack
# for serving inferences in a stable way.
#
# Modernized for V3 migration: the original image pinned 2019-era packages
# (numpy==1.16.2, scipy==1.2.1, scikit-learn==0.20.2) on ubuntu:18.04, which no
# longer install with current pip. We use a maintained slim Python base and
# unpinned compatible scientific packages. The train/serve/predictor logic is
# unchanged.

FROM python:3.10-slim-bullseye

LABEL maintainer="Amazon AI <sage-learner@amazon.com>"

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Here we get all python packages.
RUN pip --no-cache-dir install numpy scipy scikit-learn pandas flask gunicorn

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
