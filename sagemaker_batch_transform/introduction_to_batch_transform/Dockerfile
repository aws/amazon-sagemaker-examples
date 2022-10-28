FROM ubuntu:16.04

MAINTAINER Amazon SageMaker Examples <amazon-sagemaker-examples@amazon.com>

RUN apt-get -y update && apt-get install -y --no-install-recommends \
    wget \
    r-base \
    r-base-dev \
    ca-certificates

RUN R -e "install.packages(c('dbscan', 'plumber'), repos='https://cloud.r-project.org')"

COPY dbscan.R /opt/ml/dbscan.R
COPY plumber.R /opt/ml/plumber.R

ENTRYPOINT ["/usr/bin/Rscript", "/opt/ml/dbscan.R", "--no-save"]
