FROM python:3.7-slim-buster

RUN pip3 install smjsindustry==1.0.0 requests botocore boto3>=1.15.0 sagemaker pandas numpy transformers typing sentencepiece nltk
RUN python3 -c "import nltk; nltk.download('punkt')"
ENV PYTHONUNBUFFERED=TRUE

ENTRYPOINT ["python3"]
