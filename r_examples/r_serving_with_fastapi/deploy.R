library(reticulate)
library(xgboost)

# explicit tell reticulate to use the system python
use_python("/usr/bin/python3")

# load our FastAPI endpoints with reticulate
source_python('endpoints.py')

# load a pretrained xgboost model
bst <- xgb.load("xgb.model")

# create a closure around our xgboost model and input data processing
inference <- function(x){
  ds <- xgb.DMatrix(data = x )
  predict(bst, ds)
}

# make our inference closure safe to send to python as a callback
safe_inference <- py_main_thread_func(inference)

# create a new FastAPI application instance
app <- make_endpoints(safe_inference)

# run our FastAPI application
run_app(app)