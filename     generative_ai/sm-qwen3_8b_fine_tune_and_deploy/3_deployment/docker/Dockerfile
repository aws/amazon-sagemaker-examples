# Use latest vLLM for Qwen3-VL support
FROM vllm/vllm-openai:latest

# Install additional dependencies for vision models
RUN pip install --no-cache-dir \
    pillow \
    qwen-vl-utils || pip install --no-cache-dir pillow

# Copy entrypoint script
COPY ./sagemaker-entrypoint.sh /app/
RUN chmod +x /app/sagemaker-entrypoint.sh

# SageMaker health check endpoint
ENV VLLM_HOST=0.0.0.0

# SageMaker requires port 8080
EXPOSE 8080

ENTRYPOINT ["/app/sagemaker-entrypoint.sh"]
