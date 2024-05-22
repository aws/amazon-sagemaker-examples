FROM nvcr.io/nvidia/pytorch:24.01-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN=true

RUN apt-get update
RUN apt-get install -y libsndfile1 ffmpeg

RUN git clone https://github.com/NVIDIA/NeMo.git /NeMo
RUN cd /NeMo && git fetch origin b0bb807081623310c8cf0ab9a3052169fd875625
RUN cd /NeMo && git reset --hard b0bb807081623310c8cf0ab9a3052169fd875625
RUN cd /NeMo && pip install -e .

RUN cd /NeMo && pip install -r /NeMo/requirements/requirements_common.txt
RUN cd /NeMo && pip install -r /NeMo/requirements/requirements_lightning.txt
RUN cd /NeMo && pip install -r /NeMo/requirements/requirements_nlp.txt

RUN pip3 install git+https://github.com/NVIDIA/Megatron-LM.git@0d983e64afcd84cab83124e0b7ca89a3d8ec9655
RUN pip3 install transformers==4.40.1
RUN pip3 install datasets==2.19.0

RUN apt-get -y install --no-install-recommends libgomp1 build-essential
RUN apt-get -y install openssh-server
RUN apt-get -y install openssh-client
RUN mkdir -p -m0755 /var/run/sshd
RUN systemctl enable ssh

RUN ssh-keygen -t rsa -b 4096 -N '' -f ~/.ssh/id_rsa
RUN cp /root/.ssh/id_rsa.pub /root/.ssh/authorized_keys
RUN echo "Host *" >> /root/.ssh/config
RUN echo "   StrictHostKeyChecking no" >> /root/.ssh/config
RUN echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config 

RUN echo "UserKnownHostsFile /dev/null" >> /etc/ssh/ssh_config && \
    sed -i 's/#\(StrictModes \).*/\1no/g' /etc/ssh/sshd_config

RUN pip install sagemaker-training

