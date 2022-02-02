FROM daskdev/dask:2022.1.1-py3.8

ENV PYTHONHASHSEED 0
ENV PYTHONIOENCODING UTF-8

RUN pip install boto3==1.20.46

RUN conda install --yes -c conda-forge \
    dask-ml \
    s3fs

# Dumb init
RUN wget -O /usr/local/bin/dumb-init https://github.com/Yelp/dumb-init/releases/download/v1.2.0/dumb-init_1.2.0_amd64
RUN chmod +x /usr/local/bin/dumb-init

RUN mkdir -p /opt/app /etc/dask
COPY dask_config/dask.yaml /etc/dask/

# Set up bootstrapping program and Spark configuration
COPY program /opt/program
RUN chmod +x /opt/program/bootstrap.py


ENTRYPOINT ["/opt/program/bootstrap.py"]
