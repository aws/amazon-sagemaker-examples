ARG AWS_REGION
ARG CPU_OR_GPU
ARG SUFFIX
ARG VERSION
ARG FRAMEWORK

# use ray 1.6.0, tensorflow 2.5.0 base docker, install Mujoco and update tensorflow to 2.7.0
FROM 462105765813.dkr.ecr.${AWS_REGION}.amazonaws.com/sagemaker-rl-ray-container:ray-1.6.0-${FRAMEWORK}-${CPU_OR_GPU}-${SUFFIX}

# build mujoco below
RUN mkdir -p /root/.mujoco \
    && wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz \
    && tar -xf mujoco.tar.gz -C /root/.mujoco \
    && rm mujoco.tar.gz

RUN wget https://www.roboti.us/file/mjkey.txt -O /root/.mujoco/mjkey.txt

ENV LD_LIBRARY_PATH /root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
        libosmesa6-dev \
        libgl1-mesa-glx \
        libglfw3 \
        patchelf \
        libglew-dev \
        libglvnd0
        
RUN apt-get remove -y libglvnd0

RUN apt-get update && apt-get install -y --no-install-recommends \
        libglvnd0 \
        libglew-dev

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        jq \
        ffmpeg \
        rsync \
        libjpeg-dev \
        libxrender1 \
        python3.7-dev \
        python3-opengl \
        wget \
        xvfb \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

RUN pip install --no-cache-dir \
    git+https://github.com/openai/mujoco-py

# https://docs.docker.com/engine/reference/builder/#understand-how-arg-and-from-interact
ARG FRAMEWORK
RUN if [ "${FRAMEWORK}" = "tf" ] ; \
        then echo 'Installing tensorflow==2.7.0 because it is compatible with numpy>1.20 which is what Mujoco needs.' \
            && pip uninstall -y tensorflow numpy \
            && pip install --no-cache-dir tensorflow==2.7.0 ; \
    fi

# https://click.palletsprojects.com/en/7.x/python3/
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Copy workaround script for incorrect hostname
COPY lib/changehostname.c /

COPY lib/start.sh /usr/local/bin/start.sh
RUN chmod +x /usr/local/bin/start.sh

# Starts framework
ENTRYPOINT ["bash", "-m", "start.sh"]

RUN pip install sagemaker-containers --upgrade

ENV PYTHONUNBUFFERED 1

############################################
# Test Installation
############################################
# Test to verify if all required dependencies installed successfully or not.
RUN python -c "import gym; import sagemaker_containers.cli.train;import mujoco_py; from sagemaker_containers.cli.train import main"

# Make things a bit easier to debug
WORKDIR /opt/ml/code

RUN pip freeze 

RUN python --version

RUN pip show ray
RUN pip show mujoco_py
RUN pip show numpy
