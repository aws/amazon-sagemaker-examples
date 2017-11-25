# Bring Your Own R Algorithm

This folder contains one notebook and several helper files:

*mars.R* and *plumber.R:* are scrips written in the R statistical language which define training and hosting functions as specified for the Amazon SageMaker, bring your own algorithm/container documentation.

*Dockerfile:* is the necessary configuration for building a docker container that calls the `mars.R` script.

*build_and_push.sh:* is a short shell script that will build and publish the algorithm's Docker container to AWS ECR.  Running `source build_and_push.sh rmars` from the shell of a system with Docker and the proper credentials will create an ECR container that aligns to `r_bring_your_own.ipynb`.

*r_bring_your_own.ipynb:* is a notebook that calls the custom container once built and pushed into ECR.
