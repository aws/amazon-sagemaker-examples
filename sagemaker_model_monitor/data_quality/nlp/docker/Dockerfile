FROM python:3.7-slim-buster

RUN pip3 install sagemaker
RUN pip3 install transformers
RUN pip3 install torch
RUN pip3 install s3fs
RUN pip3 install sentence-transformers

ENV PYTHONUNBUFFERED=TRUE

ADD evaluation.py /

ENTRYPOINT ["python3", "/evaluation.py"]