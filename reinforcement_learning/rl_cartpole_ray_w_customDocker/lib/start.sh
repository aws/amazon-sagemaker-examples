#!/usr/bin/env bash

if [ $1 == 'train' ]
then
    # Remove all nvidia gl libraries if they exists to run training in SageMaker.
    rm -rf /usr/local/nvidia/lib/libGL*
    rm -rf /usr/local/nvidia/lib/libEGL*
    rm -rf /usr/local/nvidia/lib/libOpenGL*
    rm -rf /usr/local/nvidia/lib64/libGL*
    rm -rf /usr/local/nvidia/lib64/libEGL*
    rm -rf /usr/local/nvidia/lib64/libOpenGL*

    CURRENT_HOST=$(jq .current_host  /opt/ml/input/config/resourceconfig.json)

    sed -ie "s/PLACEHOLDER_HOSTNAME/$CURRENT_HOST/g" /changehostname.c

    gcc -o /changehostname.o -c -fPIC -Wall /changehostname.c
    gcc -o /libchangehostname.so -shared -export-dynamic /changehostname.o -ldl

    LD_PRELOAD=/libchangehostname.so xvfb-run --auto-servernum -s "-screen 0 1400x900x24" train
elif [ $1 == 'serve' ]
then
    serve
fi
