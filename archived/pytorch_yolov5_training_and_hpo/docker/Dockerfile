ARG REGION
FROM 763104351884.dkr.ecr.${REGION}.amazonaws.com/pytorch-training:1.11.0-cpu-py38
RUN git clone https://github.com/ultralytics/yolov5
ENV PATH="/yolov5:${PATH}"
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3 get-pip.py
RUN pip3 install -qr /yolov5/requirements.txt
# Need this for Automated Model Tuning support
RUN pip3 install -q --no-input pyyaml
