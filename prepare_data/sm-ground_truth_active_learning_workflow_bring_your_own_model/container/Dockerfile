# Build an image that can do training and inference in SageMaker

FROM tensorflow/tensorflow:latest-py3
RUN apt-get update && \
    apt-get install -y nginx

RUN pip install gevent gunicorn flask sagemaker-containers pandas s3fs sklearn

ENV PATH="/opt/program:${PATH}"

ENV PYTHONUNBUFFERED=1

# Set up the program in the image
COPY news-classifier /opt/program
WORKDIR /opt/program

