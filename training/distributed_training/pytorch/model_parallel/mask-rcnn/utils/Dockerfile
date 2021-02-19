ARG BASE_IMAGE=mrcnn_base_image
From $BASE_IMAGE

COPY train_scripts/maskrcnn/ /maskrcnn/
RUN apt-get install -y rsync
RUN rsync -av /maskrcnn /training_results_v0.7/NVIDIA/benchmarks/maskrcnn/implementations/
COPY launch_sm.py /opt/ml/code/launch_sm.py