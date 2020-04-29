# use minimal alpine base image as we only need python and nothing else here
FROM python:2-alpine3.6

MAINTAINER Amazon SageMaker Examples <amazon-sagemaker-examples@amazon.com>

COPY train.py /train.py

ENTRYPOINT ["python2.7", "-u", "/train.py"]
