library(readr)

prefix <- '/opt/ml/'

input_path <- paste0(prefix , 'input/data/training/')
output_path <- paste0(prefix, 'output/')
model_path <- paste0(prefix, 'model/')
code_path <- paste(prefix, 'code', sep='/')
inference_code_dir <- paste(model_path, 'code', sep='/')


abalone_train <- read_csv(paste0(input_path, 'abalone_train.csv'))
abalone_valid <- read_csv(paste0(input_path, 'abalone_valid.csv'))

regressor = lm(formula = rings ~ female + male + length + diameter + height + whole_weight + shucked_weight + viscera_weight + shell_weight, data = abalone_train)
summary(regressor)

y_pred= predict(regressor, newdata=abalone_valid[,-1])
rmse <- sqrt(mean(((abalone_valid[,1] - y_pred)^2)[,]))
print(paste0("Calculated validation RMSE: ",rmse,";"))


# Save trained model
save(regressor, file = paste0(model_path,"model"))

# Save inference code to be used with model
# find the files that you want
list_of_files <- list.files(code_path)

# copy the files to the new folder
dir.create(inference_code_dir)
file.copy(list_of_files, inference_code_dir, recursive=TRUE)

print("successfully saved model & code")
