FROM ubuntu:16.04

MAINTAINER Amazon SageMaker Examples <amazon-sagemaker-examples@amazon.com>

RUN apt-get -y update && apt-get install -y --no-install-recommends \
    wget \
    r-base \
    r-base-dev \
    ca-certificates 

RUN R -e "install.packages(c('Rcpp', 'BH', 'R6', 'jsonlite', 'crayon'), repos='https://cloud.r-project.org')"

RUN wget http://cran.r-project.org/src/contrib/Archive/stringi/stringi_1.2.4.tar.gz
RUN R CMD INSTALL stringi_1.2.4.tar.gz

RUN wget http://cran.r-project.org/src/contrib/Archive/rlang/rlang_0.2.2.tar.gz
RUN R CMD INSTALL rlang_0.2.2.tar.gz

RUN wget http://cran.r-project.org/src/contrib/Archive/magrittr/magrittr_1.5.tar.gz
RUN R CMD INSTALL magrittr_1.5.tar.gz

RUN wget http://cran.r-project.org/src/contrib/Archive/later/later_0.7.5.tar.gz
RUN R CMD INSTALL later_0.7.5.tar.gz

RUN wget http://cran.r-project.org/src/contrib/Archive/promises/promises_1.0.1.tar.gz
RUN R CMD INSTALL promises_1.0.1.tar.gz

RUN wget http://cran.r-project.org/src/contrib/Archive/httpuv/httpuv_1.4.4.2.tar.gz
RUN R CMD INSTALL httpuv_1.4.4.2.tar.gz	

RUN wget http://cran.r-project.org/src/contrib/Archive/dbscan/dbscan_1.1-2.tar.gz
RUN R CMD INSTALL dbscan_1.1-2.tar.gz

RUN wget http://cran.r-project.org/src/contrib/Archive/plumber/plumber_0.4.6.tar.gz
RUN R CMD INSTALL plumber_0.4.6.tar.gz

COPY dbscan.R /opt/ml/dbscan.R
COPY plumber.R /opt/ml/plumber.R

ENTRYPOINT ["/usr/bin/Rscript", "/opt/ml/dbscan.R", "--no-save"]
