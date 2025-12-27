# Start with the official AWS Neuron vLLM image
FROM public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.11.0-neuronx-py312-sdk2.27.0-ubuntu24.04

# 1. Install Amazon Corretto 17 and build dependencies
RUN apt-get update && apt-get install -y wget gnupg software-properties-common jq curl git build-essential && \
    wget -O - https://apt.corretto.aws/corretto.key | gpg --dearmor -o /usr/share/keyrings/corretto-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/corretto-keyring.gpg] https://apt.corretto.aws stable main" | tee /etc/apt/sources.list.d/corretto.list && \
    apt-get update && apt-get install -y java-17-amazon-corretto-jdk && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 2. Set Java Environment
ENV JAVA_HOME=/usr/lib/jvm/java-17-amazon-corretto
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# 3. Clone DJL Serving and apply patches BEFORE building
WORKDIR /usr/src
COPY patches/fix_vllm_boolean.py /tmp/fix_vllm_boolean.py
RUN git clone https://github.com/deepjavalibrary/djl-serving.git && \
    cd djl-serving && \
    git checkout 0.36.0-dlc && \
    # Apply the vLLM boolean fix BEFORE building
    python3 /tmp/fix_vllm_boolean.py && \
    rm /tmp/fix_vllm_boolean.py && \
    # Build Java binaries (this will include the patched Python code in the JAR)
    ./gradlew :serving:installDist -x test && \
    # Install Python sidecar glue code
    pip install --no-cache-dir engines/python/setup/.

# 4. Setup Final Directory Structure and Copy Assets from the Clone
ENV MODEL_SERVER_HOME=/opt/djl
RUN mkdir -p /opt/djl/conf /opt/djl/bin /opt/djl/plugins /opt/djl/scripts /tmp/.djl.ai && \
    # Copy compiled Java distribution
    cp -r /usr/src/djl-serving/serving/build/install/serving/* /opt/djl/ && \
    # Symlink the binary to /usr/bin so the entrypoint script finds it
    ln -s /opt/djl/bin/serving /usr/bin/djl-serving && \
    # Copy scripts and entrypoint from the source tree
    cp /usr/src/djl-serving/serving/docker/dockerd-entrypoint.sh /usr/local/bin/dockerd-entrypoint.sh && \
    cp -r /usr/src/djl-serving/serving/docker/scripts/* /opt/djl/scripts/ && \
    # Ensure entrypoint is executable
    chmod +x /usr/local/bin/dockerd-entrypoint.sh && \
    # Clean up source to save space
    rm -rf /usr/src/djl-serving

# 5. Final Environment Configurations (Adapted for Neuron)
ENV MODEL_SERVER_HOME=/opt/djl \
    # Point DJL to the SageMaker model mount point
    MODEL_STORE=/opt/ml/model \
    # Ensure the server binds to all interfaces for SageMaker health checks
    JAVA_OPTS="-Dbuild_directory=/opt/ml/model -Dinference_address=http://0.0.0.0:8080"\
    PATH="${MODEL_SERVER_HOME}/bin:${PATH}" \
    PJRT_DEVICE=NEURON \
    SERVING_FEATURES=vllm \
    VLLM_NO_USAGE_STATS=1 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    DJL_CACHE_DIR=/tmp/.djl.ai \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HOME=/tmp/.cache/huggingface

# 6. Create SageMaker-compatible config
# 6. Create SageMaker-compatible config
RUN echo "management_address=http://0.0.0.0:8081\n\
inference_address=http://0.0.0.0:8080\n\
model_store=/opt/ml/model" > /opt/djl/conf/config.properties

# 7. Install required packages
RUN pip install --no-cache-dir --extra-index-url=https://pip.repos.neuron.amazonaws.com \
    hf_transfer torch-neuronx==2.9.0.2.11.19912+e48cd891 vllm==0.12.0

# 8. Patch vllm_neuron for vLLM 0.12.0 compatibility
RUN sed -i 's/from vllm\.utils import make_tensor_with_pad/from vllm.utils.torch_utils import make_tensor_with_pad/g' \
    /opt/vllm/vllm_neuron/worker/neuronx_distributed_model_runner.py && \
    sed -i '/block_sizes=\[self\.block_size\],/a\            kernel_block_sizes=[self.block_size],' \
    /opt/vllm/vllm_neuron/worker/neuronx_distributed_model_runner.py && \
    sed -i '/default_neuron_config_args = _get_default_neuron_config(/i\    if not hasattr(scheduler_config, "max_model_len"):\n        scheduler_config.max_model_len = model_config.max_model_len\n' \
    /opt/vllm/vllm_neuron/worker/neuronx_distributed_model_loader.py

# Use the entrypoint and command from the cloned source
ENTRYPOINT ["/usr/local/bin/dockerd-entrypoint.sh"]
CMD ["serve"]

