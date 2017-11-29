# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.


# Bring in library that contains multivariate adaptive regression splines (MARS)
library(mda)

# Bring in library that allows parsing of JSON training parameters
library(jsonlite)

# Bring in library for prediction server
library(plumber)


# Setup parameters
# Container directories
prefix <- '/opt/ml'
input_path <- paste(prefix, 'input/data', sep='/')
output_path <- paste(prefix, 'output', sep='/')
model_path <- paste(prefix, 'model', sep='/')
param_path <- paste(prefix, 'input/config/hyperparameters.json', sep='/')

# Channel holding training data
channel_name = 'train'
training_path <- paste(input_path, channel_name, sep='/')


# Setup training function
train <- function() {

    # Read in hyperparameters
    training_params <- read_json(param_path)

    target <- training_params$target

    if (!is.null(training_params$degree)) {
        degree <- as.numeric(training_params$degree)}
    else {
        degree <- 2}

    # Bring in data
    training_files = list.files(path=training_path, full.names=TRUE)
    training_data = do.call(rbind, lapply(training_files, read.csv))
    
    # Convert to model matrix
    training_X <- model.matrix(~., training_data[, colnames(training_data) != target])

    # Save factor levels for scoring
    factor_levels <- lapply(training_data[, sapply(training_data, is.factor), drop=FALSE],
                            function(x) {levels(x)})
    
    # Run multivariate adaptive regression splines algorithm
    model <- mars(x=training_X, y=training_data[, target], degree=degree)
    
    # Generate outputs
    mars_model <- model[!(names(model) %in% c('x', 'residuals', 'fitted.values'))]
    attributes(mars_model)$class <- 'mars'
    save(mars_model, factor_levels, file=paste(model_path, 'mars_model.RData', sep='/'))
    print(summary(mars_model))

    write.csv(model$fitted.values, paste(output_path, 'data/fitted_values.csv', sep='/'), row.names=FALSE)
    write('success', file=paste(output_path, 'success', sep='/'))}


# Setup scoring function
serve <- function() {
    app <- plumb(paste(prefix, 'plumber.R', sep='/'))
    app$run(host='0.0.0.0', port=8080)}


# Run at start-up
args <- commandArgs()
if (any(grepl('train', args))) {
    train()}
if (any(grepl('serve', args))) {
    serve()}
