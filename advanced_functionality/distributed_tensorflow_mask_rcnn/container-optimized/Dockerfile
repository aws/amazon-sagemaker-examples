FROM  763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:1.13-horovod-gpu-py36-cu100-ubuntu16.04-v2.0

ENV HOROVOD_VERSION=0.18.1

RUN pip install --upgrade pip

# Need to reinstall some libraries the DL container provides due to custom Tensorflow binary
RUN pip uninstall -y tensorflow tensorboard tensorflow-estimator keras h5py horovod numpy

# Download and install custom Tensorflow binary
RUN wget https://github.com/aws-samples/mask-rcnn-tensorflow/releases/download/v0.0.0/tensorflow-1.13.0-cp36-cp36m-linux_x86_64.whl && \
    pip install tensorflow-1.13.0-cp36-cp36m-linux_x86_64.whl && \
    pip install tensorflow-estimator==1.13.0 && \
    rm tensorflow-1.13.0-cp36-cp36m-linux_x86_64.whl

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
RUN pip install numpy==1.17.5

RUN git clone https://github.com/aws-samples/mask-rcnn-tensorflow
RUN cd /mask-rcnn-tensorflow && git fetch origin 153442bc70b06e59f2bbeadc4d359b240f64cbc2
RUN cd /mask-rcnn-tensorflow && git reset --hard 153442bc70b06e59f2bbeadc4d359b240f64cbc2

RUN chmod -R +w /mask-rcnn-tensorflow
RUN pip install -e /mask-rcnn-tensorflow/

##########################################################################################
# SageMaker requirements
##########################################################################################
COPY resources/train.py /opt/ml/code/train.py

ENV SAGEMAKER_PROGRAM train.py
