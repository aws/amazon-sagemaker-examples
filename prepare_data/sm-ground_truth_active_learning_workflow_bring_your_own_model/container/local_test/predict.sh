#!/bin/bash

payload=$1
content=${2:-text/csv}

curl -d @${payload} -H "Content-Type: ${content}" http://localhost:8080/invocations
