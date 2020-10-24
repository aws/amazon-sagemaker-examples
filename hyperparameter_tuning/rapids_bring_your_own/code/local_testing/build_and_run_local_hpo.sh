#!/usr/bin/env bash

# launch container with local directory paths mounted to mirror SageMaker
echo 'run RAPIDS HPO container with local directory mirroring SageMaker paths'

# --------------------------------
# decide what runs in this script
# --------------------------------

# test multiple configurations [ xgboost/rf and single/multi-cpu/gpu ]
RUN_TESTS_FLAG=true

# run HPO container in training mode
RUN_TRAINING_FLAG=true

# run HPO container in serving mode, with or without GPU inference
RUN_SERVING_FLAG=false
GPU_SERVING_FLAG=true

# --------------------------------
# directory and dataset choices
# --------------------------------

# SageMaker directory structure [ container internal] which we'll build on 
SAGEMAKER_ROOT_DIR="/opt/ml"

# path to local directory which we'll set up to mirror cloud structure
LOCAL_TEST_DIR=~/local_sagemaker

# declare location of local Parquet and/or CSV datasets
CSV_DATA=/home/m/data/NYC_taxi
PARQUET_DATA=/home/m/data/1_year_2019

# by default script runs from /cloud-ml-examples/aws/code/local_testing
CODE_PATH=../

# expand relative to full paths for docker
LOCAL_TEST_DIR=$(realpath ${LOCAL_TEST_DIR})

# clear directories before adding code
rm -rf ${LOCAL_TEST_DIR}/code/*
rm -rf ${LOCAL_TEST_DIR}/output/*

# create directory structure to replicate SageMaker
mkdir -p ${LOCAL_TEST_DIR}/code
mkdir -p ${LOCAL_TEST_DIR}/code/workflows
mkdir -p ${LOCAL_TEST_DIR}/model
mkdir -p ${LOCAL_TEST_DIR}/output
mkdir -p ${LOCAL_TEST_DIR}/input/config
mkdir -p ${LOCAL_TEST_DIR}/input/data/training

# --------------------------------
# build container
# --------------------------------

# select wich version of the RAPIDS container is used as base for HPO
if [ "$1" == "14" ]; then
    # previous
    RAPIDS_VERSION="14"
    REPO_PREFIX="rapidsai/rapidsai"
    CUDA_VERSION="10.2"
    RUNTIME_OR_BASE="base"
elif [ "$1" == "16" ]; then
    # next
    RAPIDS_VERSION="16"
    REPO_PREFIX="rapidsai/rapidsai-nightly"
    CUDA_VERSION="11.0"
    RUNTIME_OR_BASE="base"
else
    # stable [ default ]
    RAPIDS_VERSION="15"
    REPO_PREFIX="rapidsai/rapidsai"
    CUDA_VERSION="10.2"
    RUNTIME_OR_BASE="base"
fi

DOCKERFILE_NAME="Dockerfile.$RAPIDS_VERSION"

CONTAINER_IMAGE="cloud-ml-sagemaker"
CONTAINER_TAG="0.$RAPIDS_VERSION-cuda$CUDA_VERSION-$RUNTIME_OR_BASE-ubuntu18.04-py3.7"

JUPYTER_PORT="8899"

# build the container locally
echo "pull build and tag container"
sudo docker pull ${REPO_PREFIX}:${CONTAINER_TAG}
sudo docker build ${CODE_PATH} --tag ${CONTAINER_IMAGE}:${CONTAINER_TAG} -f ${CODE_PATH}local_testing/${DOCKERFILE_NAME} 

# copy custom logic into local folder
cp -r ${CODE_PATH} ${LOCAL_TEST_DIR}/code

# --------------------------------
# launch command
# --------------------------------
function launch_container {
    # train or serve
    RUN_COMMAND=${1-"train"}
    # mounted dataset choice
    LOCAL_DATA_DIR=${2:-$PARQUET_DATA}
    # configuration settings
    DATASET_DIRECTORY=${3:-"1_year"} 
    ALGORITHM_CHOICE=${4:-"xgboost"}
    ML_WORKFLOW_CHOICE=${5:-"singlegpu"}
    # GPUs en/dis-abled within container
    GPU_ENABLED_FLAG=${6:-true}

    CV_FOLDS=${7:-"1"}
    JOB_NAME="local-test"

    # select whether GPUs are enabled
    if $GPU_ENABLED_FLAG; then
        GPU_ENUMERATION="--gpus all"
    else
        GPU_ENUMERATION=""
    fi

    sudo docker run --rm -it \
                    ${GPU_ENUMERATION} \
                    -p $JUPYTER_PORT:8888 -p 8080:8080 \
                    --env SM_TRAINING_ENV='{"job_name":''"'${JOB_NAME}'"''}'\
                    --env DATASET_DIRECTORY=${DATASET_DIRECTORY} \
                    --env ALGORITHM_CHOICE=${ALGORITHM_CHOICE} \
                    --env ML_WORKFLOW_CHOICE=${ML_WORKFLOW_CHOICE} \
                    --env CV_FOLDS=${CV_FOLDS} \
                    -v ${LOCAL_TEST_DIR}:${SAGEMAKER_ROOT_DIR} \
                    -v ${LOCAL_DATA_DIR}:${SAGEMAKER_ROOT_DIR}/input/data/training \
                    --workdir ${SAGEMAKER_ROOT_DIR}/code \
                    ${CONTAINER_IMAGE}:${CONTAINER_TAG} ${RUN_COMMAND}
}


# --------------------------------
# test definitions
# --------------------------------
function test_multiple_configurations {

    # dataset
    for idataset in {1..2}
    do
        if (( $idataset==1 )); then
            DATASET_CHOICE=$PARQUET_DATA
            DATASET_DIRECTORY="1_year"
        else
            DATASET_CHOICE=$CSV_DATA
            DATASET_DIRECTORY="nyc_taxi"
        fi
        echo ${DATASET_JOB_PREFIX}

        # algorithm
        for ialgorithm in {1..2}
        do
            if (( $ialgorithm==1 )); then
                ALGORITHM_CHOICE="xgboost"
            else
                ALGORITHM_CHOICE="randomforest"
            fi

            # workfow
            for iworkflow in {1..4}
            do
                if (( $iworkflow==1 )); then
                    ML_WORKFLOW_CHOICE="singlegpu"
                    GPU_ENABLED_FLAG=true
                elif (( $iworkflow==2 )); then
                    ML_WORKFLOW_CHOICE="multigpu"
                    GPU_ENABLED_FLAG=true
                elif (( $iworkflow==3 )); then
                    ML_WORKFLOW_CHOICE="singlecpu"
                    GPU_ENABLED_FLAG=false
                elif (( $iworkflow==4 )); then
                    ML_WORKFLOW_CHOICE="multicpu"
                    GPU_ENABLED_FLAG=false
                fi

                echo -e "----------------------------------------------\n"
                echo -e " starting test "
                echo -e "----------------------------------------------\n"
                launch_container "train" $DATASET_CHOICE $DATASET_DIRECTORY $ALGORITHM_CHOICE $ML_WORKFLOW_CHOICE $GPU_ENABLED_FLAG
                echo -e "--- end of test #${iconfig} ---\n"

            done
        done
    done
    return
}

# --------------------------------
# execute selected choices
# --------------------------------

# launch container in multiple configurations
if $RUN_TESTS_FLAG; then
    test_multiple_configurations
fi

# launch container in training mode
if $RUN_TRAINING_FLAG; then
    # delete previous models if re-training
    rm -rf ${LOCAL_TEST_DIR}/model/*

    launch_container "train"
fi

# launch container in serving mode
if $RUN_SERVING_FLAG; then   
    launch_container "serve" "" "${GPU_SERVING_FLAG}"
fi

exit