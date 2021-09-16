ARG CPU_OR_GPU
ARG AWS_REGION
ARG FRAMEWORK

FROM 462105765813.dkr.ecr.${AWS_REGION}.amazonaws.com/sagemaker-rl-ray-container:ray-0.8.5-${FRAMEWORK}-${CPU_OR_GPU}-py36

############################################
# EnergyPlus
############################################

# Install EnergyPlus. Instructions borrowed from: https://github.com/NREL/docker-energyplus
# This is not ideal. The tarballs are not named nicely and EnergyPlus versioning is strange
ENV ENERGYPLUS_VERSION 9.3.0
ENV ENERGYPLUS_TAG v9.3.0
ENV ENERGYPLUS_SHA baff08990c

# This should be x.y.z, but EnergyPlus convention is x-y-z
ENV ENERGYPLUS_INSTALL_VERSION 9-3-0

# Downloading from Github
# e.g. https://github.com/NREL/EnergyPlus/releases/download/v9.3.0/EnergyPlus-9.3.0-baff08990c-Linux-x86_64.sh
ENV ENERGYPLUS_DOWNLOAD_BASE_URL https://github.com/NREL/EnergyPlus/releases/download/$ENERGYPLUS_TAG
ENV ENERGYPLUS_DOWNLOAD_FILENAME EnergyPlus-$ENERGYPLUS_VERSION-$ENERGYPLUS_SHA-Linux-x86_64.sh
ENV ENERGYPLUS_DOWNLOAD_URL $ENERGYPLUS_DOWNLOAD_BASE_URL/$ENERGYPLUS_DOWNLOAD_FILENAME

# Collapse the update of packages, download and installation into one command to make the container smaller & remove a 
# bunch of the auxiliary apps/files that are not needed in the container
RUN apt-get update && apt-get install -y ca-certificates curl \
    && curl -SLO $ENERGYPLUS_DOWNLOAD_URL \
    && chmod +x $ENERGYPLUS_DOWNLOAD_FILENAME \
    && echo "y\r" | ./$ENERGYPLUS_DOWNLOAD_FILENAME \
    && rm $ENERGYPLUS_DOWNLOAD_FILENAME \
    && cd /usr/local/EnergyPlus-$ENERGYPLUS_INSTALL_VERSION \
    && rm -rf DataSets Documentation ExampleFiles WeatherData MacroDataSets PostProcess/convertESOMTRpgm \
    PostProcess/EP-Compare PreProcess/FMUParser PreProcess/ParametricPreProcessor PreProcess/IDFVersionUpdater    

# Remove the broken symlinks
RUN cd /usr/local/bin \
    && find -L . -type l -delete
    
WORKDIR /opt/ml

# Tell sagemaker-containers where the launch point is for training job.
ENV SAGEMAKER_TRAINING_COMMAND /opt/ml/code/sagemaker-train.sh
ENV PYTHONPATH "/usr/local/EnergyPlus-${ENERGYPLUS_INSTALL_VERSION}/"

ENV PYTHONUNBUFFERED 1
