FROM python:3.6

LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true

RUN pip install crcmod flask gunicorn tensorflow

# Add flask app directory
COPY ./app /app
WORKDIR /app

# Copy entrypoint file and make it executable
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
