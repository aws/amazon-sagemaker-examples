ARG region

FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:2.3.1-gpu-py37-cu110-ubuntu18.04

RUN 	pip --no-cache-dir --no-cache install \
        Cython \
        matplotlib \
        opencv-python-headless \
        mpi4py \
        Pillow \
        pytest \
        pyyaml

RUN 	cd /root && \
	git clone https://github.com/pybind/pybind11 && \
	cd pybind11 && \
	cmake . && \
	make -j96 install && \
	pip install .

RUN 	pip --no-cache-dir --no-cache install \
    	'git+https://github.com/NVIDIA/cocoapi#egg=pycocotools&subdirectory=PythonAPI' && \
	pip --no-cache-dir --no-cache install \
    	'git+https://github.com/NVIDIA/dllogger'