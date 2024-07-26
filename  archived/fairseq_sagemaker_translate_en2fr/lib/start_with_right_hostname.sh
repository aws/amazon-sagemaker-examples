#!/usr/bin/env bash

if [ $1 == 'train' ]
then
    CURRENT_HOST=$(jq .current_host  /opt/ml/input/config/resourceconfig.json)

    sed -ie "s/PLACEHOLDER_HOSTNAME/$CURRENT_HOST/g" /changehostname.c

    gcc -o /changehostname.o -c -fPIC -Wall /changehostname.c
    gcc -o /libchangehostname.so -shared -export-dynamic /changehostname.o -ldl

    LD_PRELOAD=/libchangehostname.so train
else
    serve
fi