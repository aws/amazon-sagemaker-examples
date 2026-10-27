ARG SM_TRITON_IMAGE_URI
FROM ${SM_TRITON_IMAGE_URI}

RUN mkdir -p /mylib/udev/rules.d/
# setup.sh script from python backend picks up the rules from here.
COPY ./mylib/* /mylib/udev/rules.d/

# Install both TensorFlow and PyTorch Neuron packages
RUN git clone https://github.com/triton-inference-server/python_backend \
 && cd python_backend \
 && source inferentia/scripts/setup.sh -t -inf2 \
 && source inferentia/scripts/setup.sh -p -inf2


