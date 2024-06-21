import numpy as np
import xgboost

from sklearn.metrics import mean_squared_error


def evaluate(model, test_df):
    y_test = test_df.iloc[:, 0].to_numpy()
    test_df.drop(test_df.columns[0], axis=1, inplace=True)
    x_test = test_df.to_numpy()

    predictions = model.predict(xgboost.DMatrix(x_test))

    mse = mean_squared_error(y_test, predictions)
    std = np.std(y_test - predictions)
    report_dict = {
        "regression_metrics": {
            "mse": {"value": mse, "standard_deviation": std},
        },
    }
    print(f"evaluation report: {report_dict}")
    return report_dict
