import io
from collections import OrderedDict

import numpy as np
import pandas as pd
import shap


def force_plot(expected_value, shap_values, feature_data, feature_headers):
    """
    Visualize the given SHAP values with an additive force layout.

    For more information: https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/tree_based_models/Force%20Plot%20Colors.html
    """
    force_plot_display = shap.plots.force(
        base_value=expected_value,
        shap_values=shap_values,
        features=feature_data,
        feature_names=feature_headers,
        matplotlib=True,
    )


def display_plots(explanations, expected_value, request_records, predictions):
    """
    Display the Model Explainability plots
    """
    per_request_shap_values = OrderedDict()
    feature_headers = []
    for i, record_output in enumerate(explanations):
        per_record_shap_values = []
        if record_output is not None:
            feature_headers = []
            for feature_attribution in record_output:
                per_record_shap_values.append(
                    feature_attribution["attributions"][0]["attribution"][0]
                )
                feature_headers.append(feature_attribution["feature_header"])
            per_request_shap_values[i] = per_record_shap_values

    for record_index, shap_values in per_request_shap_values.items():
        print(
            f"Visualize the SHAP values for Record number {record_index + 1} with Model Prediction: {predictions[record_index][0]}"
        )
        force_plot(
            expected_value,
            np.array(shap_values),
            request_records.iloc[record_index],
            feature_headers,
        )


def visualize_result(result, request_records, expected_value):
    """
    Visualize the output from the endpoint.
    """
    predictions = pd.read_csv(io.StringIO(result["predictions"]["data"]), header=None)
    predictions = predictions.values.tolist()
    print(f"Model Inference output: ")
    for i, model_output in enumerate(predictions):
        print(f"Record: {i + 1}\tModel Prediction: {model_output[0]}")

    if "kernel_shap" in result["explanations"]:
        explanations = result["explanations"]["kernel_shap"]
        display_plots(explanations, expected_value, request_records, predictions)
    else:
        print(f"No Clarify explanations for the record(s)")
