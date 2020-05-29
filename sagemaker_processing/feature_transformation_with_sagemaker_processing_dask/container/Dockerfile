FROM continuumio/miniconda3:4.7.12


RUN apt-get update
RUN apt-get install -y curl unzip python3 python3-setuptools python3-pip python-dev python3-dev python-psutil
RUN pip3 install py4j psutil==5.6.5 numpy==1.17.4
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

ENV PYTHONHASHSEED 0
ENV PYTHONIOENCODING UTF-8
ENV PIP_DISABLE_PIP_VERSION_CHECK 1


RUN conda install --yes \
    -c conda-forge \
    python==3.8 \
    python-blosc \
    cytoolz \
    dask==2.16.0 \
    distributed==2.16.0 \
    lz4 \
    nomkl \
    numpy==1.18.1 \
    pandas==1.0.1 \
    tini==0.18.0 \
    && conda clean -tipsy \
    && find /opt/conda/ -type f,l -name '*.a' -delete \
    && find /opt/conda/ -type f,l -name '*.pyc' -delete \
    && find /opt/conda/ -type f,l -name '*.js.map' -delete \
    && find /opt/conda/lib/python*/site-packages/bokeh/server/static -type f,l -name '*.js' -not -name '*.min.js' -delete \
    && rm -rf /opt/conda/pkgs

RUN pip install dask-ml 
# Dumb init
RUN wget -O /usr/local/bin/dumb-init https://github.com/Yelp/dumb-init/releases/download/v1.2.0/dumb-init_1.2.0_amd64
RUN chmod +x /usr/local/bin/dumb-init
RUN apt-get update
RUN apt-get install -y curl unzip python3 python3-setuptools python3-pip python-dev python3-dev python-psutil


RUN conda install --yes s3fs -c conda-forge



RUN mkdir /opt/app /etc/dask
COPY dask_config/dask.yaml /etc/dask/
# Set up bootstrapping program and Spark configuration
COPY program /opt/program
RUN chmod +x /opt/program/bootstrap.py


ENTRYPOINT ["/opt/program/bootstrap.py"]
