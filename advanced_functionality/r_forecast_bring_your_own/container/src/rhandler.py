from rpy2 import robjects
import rpy2.robjects.packages as rpackages


class RHandler(object):

    def __init__(self):
        robjects.r('source("forecast_methods.R")')
        r_method_names = ["ets", "ets_additive", "arima", "tbats"]
        self.r_methods = {n: robjects.r[n] for n in r_method_names}
        self.stats_pkg = rpackages.importr('stats')
        self.wrappers = {n: self.get_wrapper(method) for n, method in self.r_methods.items()}


    def unlist(self, l):
        if type(l).__name__.endswith("Vector"):
            return [self.unlist(x) for x in l]
        else:
            return l


    def get_wrapper(self, method):
        make_ts = self.stats_pkg.ts 
        def wrapper(inputs, params):
            r_params = robjects.vectors.ListVector(params)
            forecasts = []
            for x in inputs:
                vec = robjects.FloatVector(x["target"])
                ts = make_ts(vec, frequency=params["frequency"])
                forecast = method(ts, r_params)
                forecast_dict = dict(zip(forecast.names, map(self.unlist, list(forecast))))
                if "quantiles" in params["output_types"]:
                    quantiles = {}
                    for i, q in enumerate(params["quantiles"]):
                        qf = float(q)
                        if qf > 0.5:
                            quantiles[q] = forecast_dict["quantiles_upper"][i]
                        else:
                            quantiles[q] = forecast_dict["quantiles_lower"][i]
                       
                    forecast_dict["quantiles"] = quantiles
                forecast_dict = {k: forecast_dict[k] for k in params["output_types"]}
                forecasts.append(forecast_dict)
            return forecasts
        return wrapper


    def predict(self, request):
        # defaults
        configuration = {
            "method": "ets",
            "output_types": ["mean"],
            "num_samples": 10,
            "quantiles": ["0.5", "0.9"],
            "prediction_length": 10,
            "frequency": 7
        }
        
        if "configuration" in request:
            configuration.update(request["configuration"])
        # convert quantiles to confidence levels
        quantiles = [x if x > 0.5 else 1 - x for x in [float(z) for z in configuration["quantiles"]]]
        configuration["levels"] = [100-2*(100-int(100*float(x))) for x in quantiles]
        # sort quantiles by corresponding level to match the order returned by R
        configuration["quantiles"] = [x for _, x in sorted(zip(configuration["levels"], configuration["quantiles"]))]

        method = self.wrappers[configuration["method"]]

        return {"predictions": method(request["instances"], configuration)}
