FROM python:2.7

LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true

RUN pip install crcmod flask tensorflow

# Add flask app directory
COPY ./app /app
WORKDIR /app

# Direct flask to the app
ENV FLASK_APP=/app/main.py

# Copy entrypoint file and make it executable
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
