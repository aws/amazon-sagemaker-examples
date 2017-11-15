# plumber.R


#' Ping to show server is there
#' @post /ping
function() {
    list(status='200', code='200')}


#' Echo the parameter that was sent in
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
