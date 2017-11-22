#!/bin/bash

curl --data-binary @payload.csv -H "Content-Type: text/csv" -v http://localhost:8080/invocations
