FROM continuumio/miniconda3:4.7.12

ENV PYTHONHASHSEED 0
ENV PYTHONIOENCODING UTF-8

RUN conda install --yes \
    dask==2.16.0 \
    distributed==2.16.0 \
    boto3==1.16.63

RUN conda install --yes -c conda-forge \
    dask-ml==1.8.0 \
    s3fs==0.5.2

# Dumb init
RUN wget -O /usr/local/bin/dumb-init https://github.com/Yelp/dumb-init/releases/download/v1.2.0/dumb-init_1.2.0_amd64
RUN chmod +x /usr/local/bin/dumb-init

RUN mkdir /opt/app /etc/dask
COPY dask_config/dask.yaml /etc/dask/

# Set up bootstrapping program and Spark configuration
COPY program /opt/program
RUN chmod +x /opt/program/bootstrap.py


ENTRYPOINT ["/opt/program/bootstrap.py"]
