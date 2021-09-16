FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:1.15.2-gpu-py36-cu100-ubuntu18.04

RUN apt update
RUN apt install -y python3-opencv
RUN /usr/bin/python3 -m pip install --upgrade pip

RUN pip install awscli
RUN pip install boto3
RUN pip install ujson==1.35
RUN pip install opencv-python==4.1.0.25
RUN pip install Cython==0.28.4
RUN pip install pycocotools==2.0.1
RUN pip install matplotlib==3.0.3
RUN pip install markdown==3.1

RUN git clone https://github.com/tensorpack/tensorpack.git /tensorpack
RUN cd /tensorpack && git fetch origin db541e8ec9edeeb12f8dcc79f3ca835412d282e1 
RUN cd /tensorpack && git reset --hard db541e8ec9edeeb12f8dcc79f3ca835412d282e1 
RUN pip install -e /tensorpack