loadNamespace("forecast")

handle.forecast.output <- function(model, params) {
    outputs = list()
    output_types = params$output_types

    if ("samples" %in% output_types) {
        outputs$samples <- lapply(
            1:params$num_samples, 
            function(n) { simulate(model, params$prediction_length) } 
        )
    }

    if("quantiles" %in% output_types) {
        f_matrix <- forecast::forecast(
            model, 
            h=params$prediction_length, 
            level=unlist(params$levels)
        )
        outputs$quantiles_upper <- split(f_matrix$upper, col(f_matrix$upper))
        outputs$quantiles_lower <- split(f_matrix$lower, col(f_matrix$lower))
    }

    if("mean" %in% output_types) {
        outputs$mean <- forecast::forecast(model, h=params$prediction_length)$mean
    }

    outputs
}


arima <- function(ts, params) {
    model <- forecast::auto.arima(ts, trace=TRUE)
    handle.forecast.output(model, params)
}

ets <- function(ts, params) {
    model <- forecast::ets(ts)
    handle.forecast.output(model, params)
}

ets_additive <- function(ts, params) {
    model <- forecast::ets(ts, additive.only=TRUE)
    handle.forecast.output(model, params)
}

tbats <- function(ts, params) {
    model <- forecast::tbats(forecast::msts(ts, seasonal.periods=c(24,168)))
    handle.forecast.output(model, params)
}

# adding different methods here is easy, e.g.:
#
# croston <- function(ts, params) {
#     model <- forecast::croston(ts)
#     handle.forecast.output(model, params)
# }
