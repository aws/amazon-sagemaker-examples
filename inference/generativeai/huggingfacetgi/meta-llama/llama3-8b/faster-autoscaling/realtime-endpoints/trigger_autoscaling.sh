#!/bin/bash

# Check if required environment variables are set
if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ] || [ -z "$AWS_SESSION_TOKEN" ] || [ -z "$AWS_REGION" ] || [ -z "$EP_NAME" ] || [ -z "$NUM_CONCURRENT_REQUESTS" ]; then
    echo "Error: Required environment variables are not set."
    exit 1
fi

echo "Installing llmperf..."
rm -rf llmperf && \
git clone https://github.com/philschmid/llmperf.git && \
uv pip install -e llmperf/

DIR="results"

if [ ! -d "$DIR" ]; then
  mkdir -p "$DIR"
  echo "Created $DIR directory."
else
  echo "$DIR directory already exists."
fi

echo "Starting benchmarking scripts on endpoint $EP_NAME ..."

start_time=$(date +%s)

MESSAGES_API=true python llmperf/token_benchmark_ray.py \
--model $EP_NAME \
--llm-api "sagemaker" \
--max-num-completed-requests 1000 \
--timeout 600 \
--num-concurrent-requests $NUM_CONCURRENT_REQUESTS \
--results-dir "results"

end_time=$(date +%s)
echo "Execution time was $((end_time - start_time)) secs."
