# plumber.R
# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with the License. A copy of the License is located at
# 
#     http://aws.amazon.com/apache2.0/
# 
# or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.


#' Ping to show server is there
#' @get /ping
function() {
    return('')}


#' Parse input and return prediction from model
#' @param req The http request sent
#' @post /invocations
function(req) {

    # Setup locations
    prefix <- '/opt/ml'
    model_path <- paste(prefix, 'model', sep='/')

    # Bring in model file and factor levels
    load(paste(model_path, 'dbscan_model.RData', sep='/'))
    sample_training_X <- training_X[sample(nrow(training_X), 5000, replace=FALSE), ]

    # Read in data
    conn <- textConnection(gsub('\\\\n', '\n', req$postBody))
    scoring_X <- parse_file(conn)
    close(conn)
    
    print('predicting...')
    
    # Generate predictions
    return(paste(predict(model, scoring_X, sample_training_X), collapse=','))}


# Helper function to parse SageMaker PCAs output format
parse_file <- function(file) {
    json <- readLines(file)
    return(data.frame(do.call(rbind, lapply(json, FUN=parse_json))))}


# Second helper function for apply
parse_json <- function(line) {
    if (validate(line)) {
        return(do.call(rbind, fromJSON(line)[['projections']][[1]]))}}
