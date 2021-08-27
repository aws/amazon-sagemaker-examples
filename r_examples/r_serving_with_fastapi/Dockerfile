FROM r-base:3.6.3

MAINTAINER Amazon SageMaker Examples <amazon-sagemaker-examples@amazon.com>

RUN apt-get -y update && apt-get install -y --no-install-recommends \
    wget \
    r-base \
    r-base-dev \
    apt-transport-https \
    ca-certificates \
    python3 python3-dev pip

RUN pip install fastapi uvicorn numpy

RUN R -e "install.packages(c('reticulate','xgboost'), repos='https://cloud.r-project.org')"

COPY endpoints.py /opt/ml/endpoints.py
COPY deploy.R /opt/ml/deploy.R
COPY xgb.model /opt/ml/xgb.model

WORKDIR /opt/ml

ENTRYPOINT ["/usr/bin/Rscript", "/opt/ml/deploy.R", "--no-save"]



