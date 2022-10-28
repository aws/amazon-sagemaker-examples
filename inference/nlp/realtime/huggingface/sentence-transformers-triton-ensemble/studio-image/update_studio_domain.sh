#!/bin/sh

echo "studio-domain-config.json"

aws sagemaker update-domain --cli-input-json file://studio-domain-config.json