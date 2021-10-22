ARG AWS_REGION
ARG CPU_OR_GPU
ARG SUFFIX

FROM 462105765813.dkr.ecr.${AWS_REGION}.amazonaws.com/sagemaker-rl-ray-container:ray-1.6.0-tf-${CPU_OR_GPU}-${SUFFIX}

RUN pip install --upgrade pip

RUN pip show ray

# Copy the modified catlog.py
COPY src/catalog.py /usr/local/lib/python3.7/site-packages/ray/rllib/models/catalog.py