ARG SM_TRITON_IMAGE_URI
FROM 301217895009.dkr.ecr.us-west-2.amazonaws.com/sagemaker-tritonserver:23.03-py3

RUN mkdir -p /mylib/udev/rules.d/
# setup.sh script from python backend picks up the rules from here.
COPY ./mylib/* /mylib/udev/rules.d/

# Install both TensorFlow and PyTorch Neuron packages
RUN git clone https://github.com/nskool/python_backend \
 && cd python_backend \
 && git checkout add_inf2_support \
 && source inferentia/scripts/setup.sh -t -inf2 \
 && source inferentia/scripts/setup.sh -p -inf2


