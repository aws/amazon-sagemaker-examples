library(readr)
library(rjson)

model_path <- "/opt/ml/processing/model/"
model_file_tar <- paste0(model_path, "model.tar.gz")
model_file <- paste0(model_path, "model")

untar(model_file_tar, exdir = "/opt/ml/processing/model")

load(model_file)

test_path <- "/opt/ml/processing/test/"
abalone_test <- read_csv(paste0(test_path, 'abalone_test.csv'))


y_pred= predict(regressor, newdata=abalone_test[,-1])
rmse <- sqrt(mean(((abalone_test[,1] - y_pred)^2)[,]))
print(paste0("Calculated validation RMSE: ",rmse,";"))

report_dict = list(
  regression_metrics = list(
    rmse= list(value= rmse, standard_deviation = NA)
  )
)

output_dir = "/opt/ml/processing/evaluation/evaluation.json"

jsonData <- toJSON(report_dict)
write(jsonData, output_dir)
