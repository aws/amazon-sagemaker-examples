#! /usr/bin/env bash
set -e

# Get the listen port from the SM env variable, otherwise default to 8080
LISTEN_PORT=${SAGEMAKER_BIND_TO_PORT:-8080}

# Set the number of gunicorn worker processes
GUNICORN_WORKER_COUNT=$(nproc)

# Start flask app
exec gunicorn -w $GUNICORN_WORKER_COUNT -b 0.0.0.0:$LISTEN_PORT main:app
