#!/bin/bash

# Define the prefix for environment variables to look for
PREFIX="SM_VLLM_"
ARG_PREFIX="--"

# Initialize an array for storing the arguments
# port 8080 required by sagemaker
ARGS=(--port 8080)

# Boolean flags that don't take values in vLLM
BOOLEAN_FLAGS="trust-remote-code enforce-eager disable-log-stats enable-prefix-caching"

# Loop through all environment variables
while IFS='=' read -r key value; do
    # Remove the prefix from the key, convert to lowercase, and replace underscores with dashes
    arg_name=$(echo "${key#"${PREFIX}"}" | tr '[:upper:]' '[:lower:]' | tr '_' '-')
    
    # Check if this is a boolean flag
    is_boolean=false
    for flag in $BOOLEAN_FLAGS; do
        if [ "$arg_name" = "$flag" ]; then
            is_boolean=true
            break
        fi
    done
    
    if [ "$is_boolean" = true ]; then
        # For boolean flags, only add the flag if value is "true"
        if [ "$value" = "true" ] || [ "$value" = "True" ] || [ "$value" = "1" ]; then
            ARGS+=("${ARG_PREFIX}${arg_name}")
        fi
    else
        # For non-boolean args, add both flag and value
        ARGS+=("${ARG_PREFIX}${arg_name}")
        if [ -n "$value" ]; then
            ARGS+=("$value")
        fi
    fi
done < <(env | grep "^${PREFIX}")

echo "Starting vLLM with args: ${ARGS[@]}"

# Pass the collected arguments to the main entrypoint
exec python3 -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
