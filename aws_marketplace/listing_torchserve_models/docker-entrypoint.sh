#!/bin/bash
set -e

if [[ "$1" = "serve" ]]; then
    shift 1
    printenv
    ls /opt
    torchserve --start --ts-config /home/model-server/config.properties
else
    echo "$@"
    echo "command must be \"serve\""
fi

# prevent docker exit
tail -f /dev/null
