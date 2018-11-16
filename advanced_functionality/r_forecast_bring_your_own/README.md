# Bring-your-own Algorithm Sample for the R ``forecast`` Package

This example shows how to package an algorithm written in R for use with SageMaker,
by utilizing a simple Python wrapper using [rpy2](https://rpy2.bitbucket.io/).
By using Python to provide the webserver called by Sagemaker hosting,
we are able to distribute requests to all available cores (something that is
not possible using the [plumber](https://www.rplumber.io)-based [pure R
bring-your-own container](../r_bring_your_own) example).

By wrapping the powerful, automatic forecasting techniques available in the
R [forecast](https://cran.r-project.org/web/packages/forecast/index.html) package
in a SageMaker endpoint, we make them available outside of R, and by utilizing
SageMaker's ability to scale across multiple instances, we can provide the throughput
required for large-scale forecasting applications.

The overall structure and the non-R-specific code in this example are derived from the  
[scikit_bring_your_own](https://github.com/awslabs/amazon-sagemaker-examples/tree/master/advanced_functionality/scikit_bring_your_own) 
example, which contains further information on the general setup.

The algorithms we are deploying here are "hosting-only", i.e. they don't require a
separate training step, so that we only need to implement the "serve" command in this
Docker image.

There are only three files here that are substantially different from the 
[scikit_bring_your_own](https://github.com/awslabs/amazon-sagemaker-examples/tree/master/advanced_functionality/scikit_bring_your_own) 
example:

* [Dockerfile](container/Dockerfile):
  Installs all requires dependencies, including R, the 
  [forecast](https://cran.r-project.org/web/packages/forecast/index.html) 
  package, Python 3, and the [rpy2](https://rpy2.bitbucket.io) package into an 
  Ubuntu base image.
* [src/rhandler.py](container/src/rhandler.py):
  Glue code for interfacing with R using Python via ``rpy2``. Also handles some
  pre/post-processing of the data.
* [src/forecast_methods.R](container/src/forecast_methods.R): 
  R code containing simple wrapper functions for the forecasting methods
  ``ets``, ``auto.arima``, and ``tbats`` from the
  [forecast](https://cran.r-project.org/web/packages/forecast/index.html) package.

NOTE: To use this example, you will install the R forecast package as part of the
container build (using R -e 'install.packages(c("forecast"), repos="https://cloud.r-project.org") as part of the [Dockerfile](container/Dockerfile). The R forecast package is [GPL-3 licenced](https://cran.r-project.org/web/licenses/GPL-3).
