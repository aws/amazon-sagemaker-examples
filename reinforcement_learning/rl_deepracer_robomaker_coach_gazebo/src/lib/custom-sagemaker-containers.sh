#!/bin/bash

set -e
set -x

echo "Fetching leopd custom version of sagemaker-containers"

cd /tmp
rm -rf install-sagemaker-containers
mkdir install-sagemaker-containers
cd install-sagemaker-containers

git clone https://github.com/leopd/sagemaker-containers
cd sagemaker-containers
python setup.py bdist_wheel
pip install --no-cache dist/*.whl
