FROM public.ecr.aws/neuron/pytorch-training-neuronx:2.1.2-neuronx-py310-sdk2.20.2-ubuntu20.04

RUN apt-get -y update 
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
