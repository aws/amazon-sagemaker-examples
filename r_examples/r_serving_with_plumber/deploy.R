library(xgboost)
library(plumber)
library(jsonlite)

# load a pretrained xgboost model
bst <- xgb.load("xgb.model")

# create a closure around our xgboost model and input data processing
inference <- function(x){
  ds <- xgb.DMatrix(data = x )
  output <- predict(bst, ds)
  list(output=output)
}

app <- plumb('endpoints.R')
app$run(host='0.0.0.0', port=8080)
