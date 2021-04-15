# Pull R Base image from Amazon Public ECR Gallery
FROM public.ecr.aws/u6k6n4j8/r-base:latest

MAINTAINER Amazon SageMaker Examples <amazon-sagemaker-examples@amazon.com>

RUN apt-get -y update && apt-get install -y --no-install-recommends \
    apt-utils \ 
    wget \
    r-base \
    r-base-dev \
    libcurl4-openssl-dev \
    apt-transport-https \
    libsodium-dev \
    ca-certificates
    
    
# install plumber and deps from https://packages.debian.org/sid/r-cran-plumber
RUN apt-get install -y --no-install-recommends \
    libjs-bootstrap \
    libjs-jquery \
    r-api-4.0 \
    r-cran-crayon \
    r-cran-httpuv \
    r-cran-jsonlite \
    r-cran-lifecycle \
    r-cran-magrittr \
    r-cran-mime \
    r-cran-promises \
    r-cran-r6 \
    r-cran-sodium \
    r-cran-stringi \
    r-cran-swagger \
    r-cran-webutils \
    r-cran-plumber
     
# install mda and deps from https://packages.debian.org/sid/r-cran-mda
RUN apt-get install -y --no-install-recommends \
    libc6 \
    libgfortran5 \
    r-cran-class \
    r-cran-testthat \
    r-cran-mda

COPY mars.R /opt/ml/mars.R
COPY plumber.R /opt/ml/plumber.R

ENTRYPOINT ["/usr/bin/Rscript", "/opt/ml/mars.R", "--no-save"]
