
FROM python:3.7-slim-buster

RUN pip3 install pandas==0.25.3 scikit-learn==0.21.3
ENV PYTHONUNBUFFERED=TRUE

RUN mkdir /code
WORKDIR /code
COPY preprocessing.py /code

ENTRYPOINT ["python3", "preprocessing.py"]
