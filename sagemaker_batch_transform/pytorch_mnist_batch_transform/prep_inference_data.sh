#!/bin/bash

SAMPLE_FOLDER=$1

# download mnist dataset in png format
wget https://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz

tar -xzf mnist_png.tar.gz

# prepare the sample folder
rm -rf $SAMPLE_FOLDER
mkdir $SAMPLE_FOLDER

# random copy 100 images per digit to the sample folder
for i in {0..9}
do
    find ./mnist_png/testing/$i/ -type f -name "*.png" -print0 | xargs -0 shuf -e -n 100 -z | xargs -0 cp -vt $SAMPLE_FOLDER
done

# clean downloaded dataset
rm mnist_png.tar.gz
rm -rf mnist_png
