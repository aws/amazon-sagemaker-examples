# Build an image that can do training and inference in SageMaker
# This is a Python 2 image that uses the nginx, gunicorn, flask stack
# for serving inferences in a stable way.

FROM ubuntu:16.04

MAINTAINER Amazon AI <sage-learner@amazon.com>


RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         python \
         nginx \
         ca-certificates \
		 libpython2.7 \
    && rm -rf /var/lib/apt/lists/*

# Here we get all python packages.
# Symbolically linking "dist-packages" as "site-packages"
# Likewise, pip leaves the install caches populated which uses
# a significant amount of space. These optimizations save a fair amount of space in the
# image, which reduces start up time.
RUN wget https://bootstrap.pypa.io/get-pip.py && python get-pip.py && \
    pip install intel-numpy intel-scipy intel-scikit-learn flask gevent gunicorn && rm -rf /root/.cache && \	
	(cd usr/local/lib/python2.7/;mv ./dist-packages/* ./site-packages/; rm -rf ./dist-packages; \
	ln -s /usr/local/lib/python2.7/site-packages /usr/local/lib/python2.7/dist-packages)
RUN pip install pandas && rm -rf /root/.cache

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

