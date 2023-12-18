FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.1.0-gpu-py310-cu118-ubuntu20.04-sagemaker

# install the needed system packages
RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python
RUN ln -s /usr/bin/pip3 /usr/bin/pip
RUN pip install torch pandas numpy flask gevent gunicorn

# copy or create the needed files
COPY /code /opt/program

# this environment variable is used by the SageMaker PyTorch container to determine our user code directory.

ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/program
ENV WORKDIR /opt/program
WORKDIR /opt/program
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"
ENTRYPOINT ["python", "serve.py"]

