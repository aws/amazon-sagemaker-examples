FROM nvcr.io/nvidia/tensorflow:20.03-tf1-py3

RUN pip install --upgrade pip
RUN apt-get -y update
RUN apt-get install -y libsm6 libxext6 libxrender-dev

# Install OpenSSH for MPI to communicate between containers
RUN apt-get install -y --no-install-recommends openssh-client openssh-server
RUN mkdir -p /var/run/sshd && \
  sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

RUN rm -rf /root/.ssh/ && \
  mkdir -p /root/.ssh/ && \
  ssh-keygen -q -t rsa -N '' -f /root/.ssh/id_rsa && \
  cp /root/.ssh/id_rsa.pub /root/.ssh/authorized_keys && \
  printf "Host *\n  StrictHostKeyChecking no\n" >> /root/.ssh/config

# Here we get all python packages.
RUN pip install awscli
RUN pip install boto3
RUN pip install ujson
RUN pip install opencv-python

RUN git clone https://github.com/aws-samples/mask-rcnn-tensorflow /mask-rcnn-tensorflow
RUN cd /mask-rcnn-tensorflow && git fetch origin 153442bc70b06e59f2bbeadc4d359b240f64cbc2
RUN cd /mask-rcnn-tensorflow && git reset --hard 153442bc70b06e59f2bbeadc4d359b240f64cbc2

RUN chmod -R +w /mask-rcnn-tensorflow
RUN pip install -e /mask-rcnn-tensorflow/

##########################################################################################
# SageMaker requirements

# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.
##########################################################################################

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE

RUN pip install sagemaker-containers
COPY resources/train.py /opt/ml/code/train.py
RUN chmod -R +x /opt/ml/

WORKDIR /opt/ml/code/
ENV PATH="/opt/ml/code:${PATH}"
ENV SAGEMAKER_PROGRAM train.py

