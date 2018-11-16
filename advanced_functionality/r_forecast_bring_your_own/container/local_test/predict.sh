#!/bin/bash

payload=$1
content_type=${2:-application/json}

curl --data-binary @${payload} -H "Content-Type: ${content_type}" -v http://localhost:8080/invocations
