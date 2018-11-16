FROM ubuntu:18.04

ARG DEBIAN_FRONTEND=noninteractive

RUN \
     apt-get update \
  && apt-get install -y --no-install-recommends \
          nginx \
          ca-certificates \
          libcurl4-openssl-dev \
          libnlopt-dev \
          lsb-release \
          libcairo-dev \
          libedit-dev \
          libxml2-dev \
          python3 \
          python3-dev \
          python3-pip \
          r-base \
          r-base-dev

RUN R -e 'install.packages(c("forecast"), repos="https://cloud.r-project.org")'

RUN \
     python3 -m pip --no-cache-dir install pip --upgrade \
  && python3 -m pip --no-cache-dir install setuptools --upgrade \
  && python3 -m pip --no-cache-dir install wheel --upgrade \
  && python3 -m pip --no-cache-dir install jinja2 numpy pandas pytest sphinx tzlocal rpy2 gunicorn flask \
  && rm -rf /root/.cache
  
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
COPY src /opt/program
WORKDIR /opt/program
