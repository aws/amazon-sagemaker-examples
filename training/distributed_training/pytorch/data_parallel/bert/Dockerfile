ARG region

FROM 763104351884.dkr.ecr.${region}.amazonaws.com/pytorch-training:1.8.1-gpu-py36-cu111-ubuntu18.04

ARG WORK_DIR="apex_build"
RUN pip --no-cache-dir --no-cache install h5py boto3 'git+https://github.com/NVIDIA/dllogger' tqdm requests; \
      cd $WORK_DIR; \
      git clone https://github.com/NVIDIA/apex; cd apex; \
      python setup.py install --cuda_ext --cpp_ext; \
      cd ../..; rm -rf $WORK_DIR;
