FROM ubuntu:20.04

MAINTAINER Amazon SageMaker Examples <amazon-sagemaker-examples@amazon.com>

# Don't prompt for tzdata on new versions of Ubuntu:
ARG DEBIAN_FRONTEND=noninteractive

# TODO: Pin versions of R and packages?
RUN apt-get -y update && apt-get install -y --no-install-recommends \
    wget \
    r-base \
    r-base-dev \
    ca-certificates

# TODO: Check success of installation (failure only warns with zero exit code by default)
# https://stackoverflow.com/questions/26244530/how-do-i-make-install-packages-return-an-error-if-an-r-package-cannot-be-install
RUN R -e "install.packages(c('mda', 'plumber'), repos='https://cloud.r-project.org')"

COPY mars.R /opt/ml/mars.R
COPY plumber.R /opt/ml/plumber.R

ENTRYPOINT ["/usr/bin/Rscript", "/opt/ml/mars.R", "--no-save"]
