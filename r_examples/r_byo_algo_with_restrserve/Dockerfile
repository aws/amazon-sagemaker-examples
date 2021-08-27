FROM r-base:3.6.3

MAINTAINER Amazon SageMaker Examples <amazon-sagemaker-examples@amazon.com>


RUN R -e "install.packages(c('RestRserve','xgboost','dplyr'), repos='https://cloud.r-project.org')"

COPY xgb.model /opt/ml/xgb.model
COPY restrserve.R /opt/ml/restrserve.R

WORKDIR /opt/ml

ENTRYPOINT ["/usr/bin/Rscript", "/opt/ml/restrserve.R", "--no-save"]




