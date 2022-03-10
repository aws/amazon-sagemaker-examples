library(plumber)
library(readr)
library(jsonlite)

# load the trained model
prefix <- '/opt/ml/'
model_path <- paste0(prefix, 'model/model')
code_path <- paste0(prefix, 'model/code/')

load(model_path)
print("Loaded model successfully")

# function to use our model. You may require to transform data to make compatible with model
inference <- function(x){
  data = read_csv(x)
  output <- predict(regressor, newdata=data)
  list(output=output)
}

app <- plumb(paste0(code_path,'endpoints.R'))
app$run(host='0.0.0.0', port=8080)
