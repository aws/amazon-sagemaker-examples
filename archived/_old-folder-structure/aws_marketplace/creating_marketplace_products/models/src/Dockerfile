FROM jupyter/scipy-notebook

# Install dependencies
USER 0
RUN apt-get update && apt-get install -y curl;
USER 1000

# Install flask server
RUN pip install -U flask gunicorn;

# Copy scoring logic and model artifacts into the docker image
COPY scoring_logic.py /scoring_logic.py
COPY wsgi.py /wsgi.py
COPY model-artifacts.joblib /opt/ml/model/model-artifacts.joblib
COPY --chown=1000:100 serve /opt/program/serve

RUN chmod 755 /opt/program/serve
ENV PATH=/opt/program:${PATH}

WORKDIR /
