library(xgboost)
library(RestRserve)

# load a pretrained xgboost model
bst <- xgb.load("xgb.model")

# create a closure around our xgboost model and input data processing
inference <- function(x){
  ds <- xgb.DMatrix(data = x )
  predict(bst, ds)
}

app = Application$new()

app$add_get(
  path = "/ping",
  FUN = function(request, response) {
    response$set_body(list(Status = "Alive"))
  })

app$add_post(
  path = "/invocations",
  FUN = function(request, response) {
    result = list(outputs = inference(do.call(rbind,request$body$features)))
    response$set_content_type("application/json")
    response$set_body(result)
  })


backend = BackendRserve$new()
backend$start(app, http_port = 8080)
