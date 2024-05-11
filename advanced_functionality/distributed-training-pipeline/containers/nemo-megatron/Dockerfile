FROM nvcr.io/nvidia/pytorch:24.04-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN=true

RUN apt-get update
RUN apt-get install -y libsndfile1 ffmpeg

RUN pip3 install git+https://github.com/NVIDIA/Megatron-LM.git@0d983e64afcd84cab83124e0b7ca89a3d8ec9655

RUN git clone https://github.com/NVIDIA/NeMo.git /NeMo && \
    cd /NeMo && \
    git fetch origin b0bb807081623310c8cf0ab9a3052169fd875625 && \
    git reset --hard b0bb807081623310c8cf0ab9a3052169fd875625 && \
    pip install .

RUN for f in $(ls /NeMo/requirements/requirements*.txt); do pip3 install --disable-pip-version-check --no-cache-dir -r $f; done

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

