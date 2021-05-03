FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:1.15.2-gpu-py36-cu100-ubuntu18.04

RUN apt update
RUN apt install -y python3-opencv
RUN pip install --upgrade pip

RUN pip install numpy==1.17.5
RUN pip install awscli
RUN pip install boto3
RUN pip install ujson==1.35
RUN pip install opencv-python==4.1.0.25
RUN pip install Cython==0.28.4
RUN pip install matplotlib==3.0.3
RUN pip install markdown==3.1
RUN pip install numba==0.38.0

RUN pip uninstall pycocotools
RUN pip install pybind11==2.5.0
RUN pip install scikit-image==0.17.2
RUN git clone https://github.com/NVIDIA/cocoapi
RUN cd cocoapi && git fetch origin 6ac4a93058202603f36fd1ce47228e7d81119e5a
RUN cd cocoapi && git reset --hard 6ac4a93058202603f36fd1ce47228e7d81119e5a
RUN cd cocoapi/PythonAPI && make install
RUN pip install -U llvmlite==0.32.1

RUN git clone https://github.com/aws-samples/mask-rcnn-tensorflow
RUN cd /mask-rcnn-tensorflow && git fetch origin 99dda649eb39874423eeaf0a9e876570e9b2f7da
RUN cd /mask-rcnn-tensorflow && git reset --hard 99dda649eb39874423eeaf0a9e876570e9b2f7da

RUN chmod -R +w /mask-rcnn-tensorflow
RUN pip install -e /mask-rcnn-tensorflow/