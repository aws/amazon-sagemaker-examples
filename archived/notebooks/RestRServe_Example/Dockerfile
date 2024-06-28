FROM r-base:3.6.3

MAINTAINER Amazon SageMaker Examples <amazon-sagemaker-examples@amazon.com>

RUN R -e "install.packages(c('RestRserve','data.table', 'stringi', 'dplyr'), repos='https://cloud.r-project.org')"
RUN wget http://cran.r-project.org/src/contrib/Archive/xgboost/xgboost_1.4.1.1.tar.gz
RUN R CMD INSTALL xgboost_1.4.1.1.tar.gz

COPY xgb.model /opt/ml/xgb.model
COPY restrserve.R /opt/ml/restrserve.R

WORKDIR /opt/ml

ENTRYPOINT ["/usr/bin/Rscript", "/opt/ml/restrserve.R", "--no-save"]
