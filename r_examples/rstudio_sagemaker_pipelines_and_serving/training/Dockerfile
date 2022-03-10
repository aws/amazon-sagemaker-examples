#FROM public.ecr.aws/docker/library/r-base:4.1.2
FROM 940823527478.dkr.ecr.eu-west-1.amazonaws.com/sagemaker-r-model-abalone:base

#RUN apt-get -y update && apt-get install -y --no-install-recommends \
#    wget \
#    apt-transport-https \
#    ca-certificates \
#    libcurl4-openssl-dev \
#    libsodium-dev
    
#RUN R -e "install.packages(c('readr','plumber'))"

#ENV PATH="/opt/ml/code:${PATH}"

COPY /code /opt/ml/code

RUN /bin/bash -c 'chmod +x /opt/ml/code/train'
RUN /bin/bash -c 'chmod +x /opt/ml/code/serve'

WORKDIR /opt/ml/code