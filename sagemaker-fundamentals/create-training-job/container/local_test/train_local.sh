#!/bin/sh

image=$1

mkdir -p test_dir/model
mkdir -p test_dir/output

rm test_dir/model/*
rm test_dir/output/*

#The --rm options tells docker run command to remove the container when it exits automatically
docker run -v $(pwd)/test_dir:/opt/ml --rm ${image} train
