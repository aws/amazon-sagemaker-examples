# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.


# Bring in library that contains multivariate adaptive regression splines (MARS)
library(dbscan)

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

    # Bring in data
    training_files <- list.files(path=training_path, full.names=TRUE)
    training_X <- do.call(rbind, lapply(training_files, FUN=parse_file))

    # Read in hyperparameters
    training_params <- read_json(param_path)

    if (!is.null(training_params$minPts)) {
        minPts <- as.numeric(training_params$minPts)}
    else {
        minPts <- 5}
    

    if (!is.null(training_params$eps)) {
        eps <- as.numeric(training_params$eps)}
    else {
        eps <- mean(kNNdist(training_X, k=minPts))}
    
    # Run DBSCAN algorithm
    model <- dbscan(training_X, eps=eps, minPts=minPts)
    print(model)

    # Generate outputs
    save(model, training_X, file=paste(model_path, 'dbscan_model.RData', sep='/'))
    write('success', file=paste(output_path, 'success', sep='/'))}
    

# Helper function to parse SageMaker PCAs output format
parse_file <- function(file) {
    json <- readLines(file)
    return(data.frame(do.call(rbind, lapply(json, FUN=parse_json))))}


# Second helper function for apply
parse_json <- function(line) {
    if (validate(line)) {
        return(do.call(rbind, fromJSON(line)[['projections']][[1]]))}}


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
