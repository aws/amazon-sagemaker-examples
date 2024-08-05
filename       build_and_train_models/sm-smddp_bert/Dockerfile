ARG dlc_account_id
ARG region

FROM ${dlc_account_id}.dkr.ecr.${region}.amazonaws.com/pytorch-training:1.10.2-gpu-py38-cu113-ubuntu20.04-sagemaker

ARG WORK_DIR="apex_build"
RUN pip --no-cache-dir --no-cache install h5py boto3 'git+https://github.com/NVIDIA/dllogger' tqdm requests; \
      cd $WORK_DIR; \
      git clone https://github.com/NVIDIA/apex; cd apex; \
      python setup.py install --cuda_ext --cpp_ext; \
      cd ../..; rm -rf $WORK_DIR;
