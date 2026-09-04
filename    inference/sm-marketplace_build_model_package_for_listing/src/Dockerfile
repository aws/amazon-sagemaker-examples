FROM jupyter/scipy-notebook

# Install dependencies
USER 0
RUN apt-get update && apt-get install -y curl;
USER 1000

# Install flask server. Upgrade the scientific stack together so the serving
# environment can unpickle the model artifact, which is trained/serialized by a
# modern kernel (numpy 2.x). The stale base-image numpy (1.24) otherwise fails
# with "No module named 'numpy._core'" when loading the joblib artifact. All of
# numpy/scipy/pandas/scikit-learn are upgraded in a single resolve to keep their
# compiled ABIs consistent (partial upgrades trigger "numpy.dtype size changed").
RUN pip install -U flask gunicorn "numpy>=2.0,<3" "scipy>=1.13" "pandas>=2.2" "scikit-learn>=1.4" joblib;

# Copy scoring logic and model artifacts into the docker image
COPY scoring_logic.py /scoring_logic.py
COPY wsgi.py /wsgi.py
COPY model-artifacts.joblib /opt/ml/model/model-artifacts.joblib
COPY --chown=1000:100 serve /opt/program/serve

RUN chmod 755 /opt/program/serve
ENV PATH=/opt/program:${PATH}

WORKDIR /
