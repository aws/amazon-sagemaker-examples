#! /usr/bin/env bash
set -e

# Get the listen port from the SM env variable, otherwise default to 8080
LISTEN_PORT=${SAGEMAKER_BIND_TO_PORT:-8080}

# Start flask app
exec flask run --host=0.0.0.0 --port=$LISTEN_PORT
