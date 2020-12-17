FROM tensorflow/tensorflow:2.2.0rc2-py3-jupyter

# Install Amazon SageMaker training toolkit and smdebug libraries
RUN pip install sagemaker-training
RUN pip install smdebug

# Copies the training code inside the container
COPY tf_keras_resnet_byoc.py /opt/ml/code/tf_keras_resnet_byoc.py

# Defines train.py as script entrypoint
ENV SAGEMAKER_PROGRAM tf_keras_resnet_byoc.py