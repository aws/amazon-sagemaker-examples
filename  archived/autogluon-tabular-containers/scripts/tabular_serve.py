from autogluon.tabular import TabularPredictor
from io import BytesIO, StringIO

import pandas as pd

from autogluon.core.constants import REGRESSION
from autogluon.core.utils import get_pred_from_proba_df


def model_fn(model_dir):
    """loads model from previously saved artifact"""
    model = TabularPredictor.load(model_dir)
    globals()["column_names"] = model.feature_metadata_in.get_features()
    return model


def transform_fn(
    model, request_body, input_content_type, output_content_type="application/json"
):
    if input_content_type == "application/x-parquet":
        buf = BytesIO(request_body)
        data = pd.read_parquet(buf)

    elif input_content_type == "text/csv":
        buf = StringIO(request_body)
        data = pd.read_csv(buf)

    elif input_content_type == "application/json":
        buf = StringIO(request_body)
        data = pd.read_json(buf)

    elif input_content_type == "application/jsonl":
        buf = StringIO(request_body)
        data = pd.read_json(buf, orient="records", lines=True)

    else:
        raise ValueError(f"{input_content_type} input content type not supported.")

    if model.problem_type != REGRESSION:
        pred_proba = model.predict_proba(data, as_pandas=True)
        pred = get_pred_from_proba_df(pred_proba, problem_type=model.problem_type)
        pred_proba.columns = [str(c) + "_proba" for c in pred_proba.columns]
        pred.name = str(pred.name) + "_pred" if pred.name is not None else "pred"
        prediction = pd.concat([pred, pred_proba], axis=1)
    else:
        prediction = model.predict(data, as_pandas=True)
    if isinstance(prediction, pd.Series):
        prediction = prediction.to_frame()

    if "application/x-parquet" in output_content_type:
        prediction.columns = prediction.columns.astype(str)
        output = prediction.to_parquet()
        output_content_type = "application/x-parquet"
    elif "application/json" in output_content_type:
        output = prediction.to_json()
        output_content_type = "application/json"
    elif "text/csv" in output_content_type:
        output = prediction.to_csv(index=None)
        output_content_type = "text/csv"
    else:
        raise ValueError(f"{output_content_type} content type not supported")

    return output, output_content_type
