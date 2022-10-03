ARG dlc_account_id
ARG region

FROM ${dlc_account_id}.dkr.ecr.${region}.amazonaws.com/pytorch-training:1.10.2-gpu-py38-cu113-ubuntu20.04-sagemaker

RUN pip install ninja yacs cython matplotlib tqdm opencv-python pybind11==2.5.0 'git+https://github.com/NVIDIA/dllogger'

RUN pip --no-cache-dir --no-cache install 'git+https://github.com/NVIDIA/cocoapi#egg=pycocotools&subdirectory=PythonAPI'

RUN 	cd /root && \
	rm -rf apex && \
	git clone https://github.com/NVIDIA/apex && \
	cd apex && \
	python setup.py install --cuda_ext --cpp_ext

RUN 	cd /root && \
	rm -rf DeepLearningExamples && \
	git clone --recursive https://github.com/HerringForks/DeepLearningExamples.git && \
	cd DeepLearningExamples/PyTorch/Segmentation/MaskRCNN/pytorch/ && \
	python setup.py build develop
