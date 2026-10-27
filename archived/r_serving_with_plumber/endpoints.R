
#' Ping to show server is there
#' @get /ping
function() {
    return('Alive')
}


#' Parse input and return prediction from model
#' @param req The http request sent
#' @post /invocations
function(req) {

    # Read in data
    input_json <- fromJSON(req$postBody)
    output <- inference(input_json$features)
    # Return prediction
    return(output)

}
