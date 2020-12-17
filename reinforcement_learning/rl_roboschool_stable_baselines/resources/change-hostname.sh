#!/usr/bin/env bash

sed -ie "s/PLACEHOLDER_HOSTNAME/$1/g" /changehostname.c

gcc -o /changehostname.o -c -fPIC -Wall /changehostname.c
gcc -o /libchangehostname.so -shared -export-dynamic /changehostname.o -ldl
