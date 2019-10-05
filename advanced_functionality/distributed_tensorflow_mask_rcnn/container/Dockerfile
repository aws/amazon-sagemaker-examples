FROM  763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:1.13-horovod-gpu-py36-cu100-ubuntu16.04-v2.0

ENV TENSORFLOW_VERSION=1.13.2
ENV HOROVOD_VERSION=0.16.4

RUN pip install --upgrade pip

RUN pip install tensorflow-gpu==${TENSORFLOW_VERSION} keras h5py

# Install Horovod, temporarily using CUDA stubs
RUN ldconfig /usr/local/cuda-10.0/targets/x86_64-linux/lib/stubs && \
    HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_WITH_TENSORFLOW=1 pip install --no-cache-dir horovod==${HOROVOD_VERSION} && \
    ldconfig

# Install OpenSSH for MPI to communicate between containers
RUN apt-get install -y --no-install-recommends openssh-client openssh-server
RUN mkdir -p /var/run/sshd && \
  sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

RUN rm -rf /root/.ssh/ && \
  mkdir -p /root/.ssh/ && \
  ssh-keygen -q -t rsa -N '' -f /root/.ssh/id_rsa && \
  cp /root/.ssh/id_rsa.pub /root/.ssh/authorized_keys && \
  printf "Host *\n  StrictHostKeyChecking no\n" >> /root/.ssh/config

RUN pip install awscli
RUN pip install boto3
RUN pip install ujson==1.35
RUN pip install opencv-python==4.1.0.25
RUN pip install Cython==0.28.4
RUN pip install pycocotools==2.0.0
RUN pip install matplotlib==3.0.3
RUN pip install markdown==3.1

RUN git clone https://github.com/tensorpack/tensorpack.git /tensorpack
RUN cd /tensorpack && git fetch origin 26664c3f1d58ae029ea6c3ba0af6ae11900b1e55 
RUN cd /tensorpack && git reset --hard 26664c3f1d58ae029ea6c3ba0af6ae11900b1e55 
RUN pip install --ignore-installed -e /tensorpack

##########################################################################################
# SageMaker requirements
##########################################################################################
COPY resources/train.py /opt/ml/code/train.py

ENV SAGEMAKER_PROGRAM train.py
