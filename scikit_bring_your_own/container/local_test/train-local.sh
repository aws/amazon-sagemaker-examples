#!/bin/sh

mkdir -p test-dir/model
mkdir -p test-dir/output

rm test-dir/model/*
rm test-dir/output/*

docker run -v $(pwd)/test-dir:/opt/ml --rm decision-trees-2 train
