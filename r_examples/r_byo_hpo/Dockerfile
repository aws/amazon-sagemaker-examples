FROM ubuntu:16.04

MAINTAINER Amazon SageMaker Examples <amazon-sagemaker-examples@amazon.com>

RUN apt-get -y update && apt-get install -y --no-install-recommends \
    wget \
    r-base \
    r-base-dev \
    apt-transport-https \
    ca-certificates

RUN R -e "install.packages('plumber', repos='https://cloud.r-project.org')"
RUN R -e "install.packages('https://cran.r-project.org/src/contrib/Archive/mda/mda_0.4-10.tar.gz', repos = NULL, type='source')"


COPY mars.R /opt/ml/mars.R
COPY plumber.R /opt/ml/plumber.R

ENTRYPOINT ["/usr/bin/Rscript", "/opt/ml/mars.R", "--no-save"]
