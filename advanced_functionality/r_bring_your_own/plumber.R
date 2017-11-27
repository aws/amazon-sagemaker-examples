# plumber.R
# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
    load(paste(model_path, 'mars_model.RData', sep='/'))

    # Read in data
    conn <- textConnection(gsub('\\\\n', '\n', req$postBody))
    data <- read.csv(conn)
    close(conn)

    # Convert input to model matrix
    scoring_X <- model.matrix(~., data, xlev=factor_levels)

    # Return prediction
    return(paste(predict(mars_model, scoring_X, row.names=FALSE), collapse=','))}
