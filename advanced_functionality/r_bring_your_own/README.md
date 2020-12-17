# Bring Your Own R Algorithm

This folder contains one notebook and several helper files:

*mars.R* and *plumber.R:* are scripts written in the R statistical language which define training and hosting functions as specified for the Amazon SageMaker, bring your own algorithm/container documentation.

*Dockerfile:* is the necessary configuration for building a docker container that calls the `mars.R` script.

*r_bring_your_own.ipynb:* is a notebook that calls the custom container once built and pushed into ECR.
