#!/bin/sh

image=$1

docker run -p 8080:8080 --rm ${image} serve
