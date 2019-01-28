ARG CPU_OR_GPU
ARG AWS_REGION
FROM 520713654638.dkr.ecr.${AWS_REGION}.amazonaws.com/sagemaker-rl-tensorflow:ray0.5.3-${CPU_OR_GPU}-py3

WORKDIR /opt/ml

############################################
# Roboschool
############################################

RUN apt-get update && apt-get install -y \
      git cmake ffmpeg pkg-config \
      qtbase5-dev libqt5opengl5-dev libassimp-dev \
      libpython3.5-dev libtinyxml-dev \
    && cd /opt \
    && git clone --depth 1 https://github.com/olegklimov/bullet3 -b roboschool_self_collision \
    && git clone --depth 1 https://github.com/openai/roboschool \
    && apt-get clean && rm -rf /var/cache/apt/archives/* /var/lib/apt/lists/*

ENV ROBOSCHOOL_PATH /opt/roboschool/

RUN mkdir -p /opt/bullet3/build \
    && cd /opt/bullet3/build \
    && cmake -DBUILD_SHARED_LIBS=ON -DUSE_DOUBLE_PRECISION=1 \
       -DCMAKE_INSTALL_PREFIX:PATH=${ROBOSCHOOL_PATH}/roboschool/cpp-household/bullet_local_install \
       -DBUILD_CPU_DEMOS=OFF -DBUILD_BULLET2_DEMOS=OFF \
       -DBUILD_EXTRAS=OFF  -DBUILD_UNIT_TESTS=OFF \
       -DBUILD_CLSOCKET=OFF -DBUILD_ENET=OFF \
       -DBUILD_OPENGL3_DEMOS=OFF .. \
    && make -j4\
    && make install \
    && cd ../..

RUN apt-get update && \
    apt-get install -y libboost-python-dev

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3.6-dev \
    && ln -s -f /usr/bin/python3.6 /usr/bin/python \
    && apt-get clean \
    && rm -rf /var/cache/apt/archives/* /var/lib/apt/lists/*

RUN curl -fSsL -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

RUN pip install --upgrade \
    pip \
    setuptools

RUN pip install -e ${ROBOSCHOOL_PATH} \
    && ldconfig
    
RUN pip install sagemaker-containers --upgrade

ENV PYTHONUNBUFFERED 1

############################################
# Test Installation
############################################
# Test to verify if all required dependencies installed successfully or not.
RUN python -c "import gym;import sagemaker_containers.cli.train; import roboschool; import ray; from sagemaker_containers.cli.train import main"

# Make things a bit easier to debug
WORKDIR /opt/ml/code
