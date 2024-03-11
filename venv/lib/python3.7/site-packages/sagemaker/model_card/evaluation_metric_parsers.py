# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""Evaluation metrics parser classes."""
from __future__ import absolute_import

from abc import ABC, abstractmethod
from enum import Enum
from collections import defaultdict
from typing import Union, List

from sagemaker.model_card.schema_constraints import (
    PYTHON_TYPE_TO_METRIC_VALUE_TYPE,
    METRIC_VALUE_TYPE_MAP,
    MetricTypeEnum,
)


class EvaluationMetricTypeEnum(str, Enum):
    """Model card evaluation metric type"""

    MODEL_CARD_METRIC_SCHEMA = "Model Card Metric Schema"
    CLARIFY_BIAS = "Clarify Bias"
    CLARIFY_EXPLAINABILITY = "Clarify Explainability"
    MODEL_MONITOR_MODEL_QUALITY = "Model Monitor Model Quality"
    REGRESSION = "Model Monitor Model Quality Regression"
    BINARY_CLASSIFICATION = "Model Monitor Model Quality Binary Classification"
    MULTICLASS_CLASSIFICATION = "Model Monitor Model Quality Multiclass Classification"


class ParserBase(ABC):
    """Base class for evaluation metrics parser."""

    @abstractmethod
    def _validate(self, json_data: dict):
        """Custom parser need to inherit from ParserBase and must supply a _validate method to test various restrictions as needed.

        Args:
            json_data (dict): Metric data to be validated.
        """  # noqa E501 # pylint: disable=line-too-long
        pass  # pylint: disable=W0107

    @abstractmethod
    def _parse(self, json_data: dict):
        """Custom parser need to inherit from ParserBase and must supply a _parse method to convert the raw data into model card format.

            e.g. {
                "metric_groups": [
                {
                    "name": "binary_classification_metrics",
                    "metric_data": []
                }
            }
        Args:
            json_data (dict): Raw metric data.
        """  # noqa E501 # pylint: disable=line-too-long
        pass  # pylint: disable=W0107

    def run(self, json_data: dict):
        """Parser entry point.

        Args:
            json_data (dict): metric data.
        """
        self._validate(json_data)
        return self._parse(json_data)


class DefaultParser(ParserBase):
    """Metric Parser for data following the model card evaluation metrics schema"""

    def _validate(self, json_data: dict):
        """Implement ParserBase._validate.

        Args:
            json_data (dict): Metric data to be validated.

        Raises:
            ValueError: Missing metric_group key.
            ValueError: metric_group object is not a list.
        """
        if "metric_groups" not in json_data:
            raise ValueError("Missing metric_group in the metric data.")
        if not isinstance(json_data["metric_groups"], list):
            raise ValueError('Invalid metric data format. {"metric_groups": [......]} is expected')

    def _parse(self, json_data: dict):
        """Implement ParserBase._parse.

        Args:
            json_data (dict): Raw metric data.
        """
        return json_data


class ClarifyBiasParser(ParserBase):
    """Metric Parser for clarify bias"""

    def _validate(self, json_data: dict):
        """Implement ParserBase._validate.

        Args:
            json_data (dict): Metric data to be validated.

        Raises:
            ValueError: Missing the facets key in the data.
        """
        for _, facets in json_data.items():
            if isinstance(facets, dict) and "facets" not in facets:
                raise ValueError(f"Missing facets value from the clarify bias {facets}.")

    def _parse(self, json_data: dict):
        """Implement ParserBase._parse.

        Args:
            json_data (dict): Raw metric data.
        """
        result = {"metric_groups": []}
        for metric_grp, facets in json_data.items():
            if isinstance(facets, dict):
                for facet_name, facet in facets["facets"].items():
                    group_data = defaultdict(list)
                    for item in facet:
                        grp_name = (
                            f"{metric_grp} - label {facets['label']} = "
                            f"{facets['label_value_or_threshold']} and "
                            f"facet {facet_name}={item['value_or_threshold']}"
                        )
                        group_data[grp_name].extend(
                            [
                                {"name": i["name"], "value": i["value"], "type": "number"}
                                for i in item["metrics"]
                                if i["value"] is not None
                            ]
                        )
                    for group_name, metric_data in group_data.items():
                        result["metric_groups"].append(
                            {"name": group_name, "metric_data": metric_data}
                        )

        return result


class ClarifyExplainabilityParser(ParserBase):
    """Metric Parser for clarify explainability"""

    def _validate(self, json_data: dict):
        """Implement ParserBase._validate.

        Args:
            json_data (dict): Metric data to be validated.

        Raises:
            ValueError: Missing explanations key.
        """
        explains = json_data.get("explanations", None)
        if not isinstance(explains, dict):
            raise ValueError("Invalid explanations value.")

    def _parse(self, json_data: dict):
        """Implement ParserBase._parse.

        Args:
            json_data (dict): Raw metric data.
        """
        metric_groups = []
        explains = json_data["explanations"]
        metric_groups.extend(self._parse_kernal_shap(explains))
        metric_groups.extend(self._parse_pdp(explains))

        return {"metric_groups": metric_groups}

    def _parse_kernal_shap(self, explains: dict):
        """Parse kernal_shap data in clarify explainability.

        Args:
            explains (dict): Explains metric data.
        """

        def format_group(group_name: str, group_data: dict):
            """Create the metric group entry.

            Args:
                group_name (str): Metric group name.
                group_data (dict): Raw metric group data.
            """
            metric_data = [
                {
                    "name": k,
                    "value": v,
                    "type": PYTHON_TYPE_TO_METRIC_VALUE_TYPE[type(v)].value,
                }
                for k, v in group_data.items()
            ]
            metric_group = {
                "name": group_name,
                "metric_data": metric_data,
            }

            return metric_group

        metric_groups = []
        if "kernel_shap" not in explains:
            return metric_groups

        for label_name, explain in explains["kernel_shap"].items():
            for key1, value1 in explain.items():
                if key1 == "global_shap_values":
                    group_name = f"explanations - {key1}, label={label_name}"
                    metric_groups.append(format_group(group_name, value1))
                if key1 == "global_top_shap_text":
                    for feature_name, feature_data in value1.items():
                        group_name = (
                            f"explanations - {key1}, label={label_name}, " f"feature={feature_name}"
                        )
                        metric_groups.append(format_group(group_name, feature_data))

        return metric_groups

    def _parse_pdp(self, explains: dict):
        """Parse pdp data in clarify explainability.

        Args:
            explains (dict): Explains metric data.
        """

        def format_metric_data(
            feature: dict,
            distribution_value: List,
            prediction_values: List,
            x_axis_name: Union[str, list],
            graph_type: str,
        ):
            """Create metric entry.

            Args:
                feature (dict): Entry in pdp list.
                distribution_value (List): y values in data distribution graph.
                prediction_values (List): y values in data prediction graph.
                x_axis_name (Union[str, list]): X axis name.
                graph_type (str): Type of graph.
            """
            metric_data = []
            metric_data.append(
                {
                    "name": f"data_distribution - feature={feature['feature_name']}",
                    "type": graph_type,
                    "value": distribution_value,
                    "x_axis_name": x_axis_name,
                    "y_axis_name": "Data distribution",
                }
            )
            for idx, label_header in enumerate(feature["label_headers"]):
                metric_data.append(
                    {
                        "name": (
                            f"model_predictions - feature={feature['feature_name']}, "
                            f"label={label_header}"
                        ),
                        "type": graph_type,
                        "value": prediction_values[idx],
                        "x_axis_name": x_axis_name,
                        "y_axis_name": label_header,
                    }
                )
            return metric_data

        metric_groups = []
        if "pdp" not in explains:
            return metric_groups

        metric_data = []
        for feature in explains["pdp"]:
            if feature["data_type"] == "numerical":
                distribution_value = list(
                    zip(feature["feature_values"], feature["data_distribution"])
                )
                prediction_values = [
                    list(zip(feature["feature_values"], feature["model_predictions"][i]))
                    for i, _ in enumerate(feature["label_headers"])
                ]
                x_axis_name = feature["feature_name"]
                metric_data.extend(
                    format_metric_data(
                        feature=feature,
                        distribution_value=distribution_value,
                        prediction_values=prediction_values,
                        x_axis_name=x_axis_name,
                        graph_type="linear_graph",
                    )
                )
            if feature["data_type"] == "categorical":
                distribution_value = feature["data_distribution"]
                prediction_values = [
                    feature["model_predictions"][i] for i, _ in enumerate(feature["label_headers"])
                ]
                x_axis_name = feature["feature_values"]
                metric_data.extend(
                    format_metric_data(
                        feature=feature,
                        distribution_value=distribution_value,
                        prediction_values=prediction_values,
                        x_axis_name=x_axis_name,
                        graph_type="bar_chart",
                    )
                )

        metric_groups.append({"name": "explanations - pdp", "metric_data": metric_data})

        return metric_groups


class ModelMonitorModelQualityParserBase(ParserBase):
    """Base metric parser for model monitor model quality"""

    def _parse_basic_metric(self, metric_name: str, metric_data: dict):
        """Parse the basic metric in the model monitor model quality.

            e.g. {
                "standard_deviation": 0.1,
                "value": 1
            }

        Args:
            metric_name (str): metric name.
            metric_data (dict): metric data.
        """
        metrics = [{"name": metric_name, "value": metric_data["value"], "type": "number"}]
        if type(metric_data["standard_deviation"]) in METRIC_VALUE_TYPE_MAP[MetricTypeEnum.NUMBER]:
            metrics.append(
                {
                    "name": f"{metric_name} - standard_deviation",
                    "value": metric_data["standard_deviation"],
                    "type": "number",
                }
            )
        return metrics


class RegressionParser(ModelMonitorModelQualityParserBase):
    """Metric parser for model monitor model quality regression data"""

    def _validate(self, json_data: dict):
        """Implement ParserBase._validate.

        Args:
            json_data (dict): Metric data to be validated.

        Raises:
            ValueError: missing regression_metrics key.
        """
        if "regression_metrics" not in json_data:
            raise ValueError("Missing regression_metrics from the metric data.")

    def _parse(self, json_data: dict):
        """Implement ParserBase._parse.

        Args:
            json_data (dict): Raw metric data.
        """
        result = {"metric_groups": []}
        for group_name, group_data in json_data.items():
            metric_data = []
            if group_name == "regression_metrics":
                for metric_name, raw_data in group_data.items():
                    metric_data.extend(self._parse_basic_metric(metric_name, raw_data))
                result["metric_groups"].append({"name": group_name, "metric_data": metric_data})
        return result


class ClassificationParser(ModelMonitorModelQualityParserBase):
    """Metric parser for model monitor model quality bianry/muliti-classification data"""

    def _validate(self, json_data: dict):
        """Implement ParserBase._validate.

        Args:
            json_data (dict): Metric data to be validated.

        Raises:
            ValueError: missing variable key.
        """
        if (
            "binary_classification_metrics" not in json_data
            and "multiclass_classification_metrics" not in json_data
        ):
            raise ValueError("Missing *_classification_metrics from the metric data.")

    def _parse(self, json_data: dict):
        """Implement ParserBase._parse.

        Args:
            json_data (dict): Raw metric data.
        """
        result = {"metric_groups": []}
        for group_name, group_data in json_data.items():
            metric_data = []
            if group_name not in (
                "binary_classification_metrics",
                "multiclass_classification_metrics",
            ):
                continue
            for metric_name, raw_data in group_data.items():
                metric_data.extend(self._parse_confusion_matrix(metric_name, raw_data))
                metric_data.extend(
                    self._parse_receiver_operating_characteristic_curve(metric_name, raw_data)
                )
                metric_data.extend(self._parse_precision_recall_curve(metric_name, raw_data))
                if metric_name not in [
                    "confusion_matrix",
                    "receiver_operating_characteristic_curve",
                    "precision_recall_curve",
                ]:
                    metric_data.extend(self._parse_basic_metric(metric_name, raw_data))
            result["metric_groups"].append({"name": group_name, "metric_data": metric_data})

            return result

    def _parse_confusion_matrix(self, metric_name, raw_data):
        """Translate the confusion matrix to a metric entry.

        Args:
            metric_name (_type_): Metric name.
            raw_data (_type_): Raw metric data.
        """
        metric_data = []
        if metric_name != "confusion_matrix":
            return metric_data

        rows = []
        x_axis_name = []
        for dim1_name, dim1 in raw_data.items():
            y_axis_name = []
            col = []
            for dim2_name, dim2 in dim1.items():
                col.append(dim2)
                y_axis_name.append(dim2_name)
            x_axis_name.append(dim1_name)
            rows.append(col)
        metric_data.extend(
            [
                {
                    "name": metric_name,
                    "type": "matrix",
                    "x_axis_name": x_axis_name,
                    "y_axis_name": y_axis_name,
                    "value": rows,
                }
            ]
        )
        return metric_data

    def _parse_receiver_operating_characteristic_curve(self, metric_name, raw_data):
        """Translate the receiver operating characteristic curve to a metric entry.

        Args:
            metric_name (_type_): Metric name.
            raw_data (_type_): Raw metric data.
        """
        metric_data = []
        if metric_name != "receiver_operating_characteristic_curve":
            return metric_data

        metric_data.extend(
            [
                {
                    "name": metric_name,
                    "type": "linear_graph",
                    "x_axis_name": "false_positive_rates",
                    "y_axis_name": "true_positive_rates",
                    "value": [
                        list(i)
                        for i in zip(
                            raw_data["false_positive_rates"],
                            raw_data["true_positive_rates"],
                        )
                    ],
                }
            ]
        )
        return metric_data

    def _parse_precision_recall_curve(self, metric_name, raw_data):
        """Translate the precision recall curve to a metric entry.

        Args:
            metric_name (_type_): Metric name.
            raw_data (_type_): Raw metric data.
        """
        metric_data = []
        if metric_name != "precision_recall_curve":
            return metric_data

        metric_data.extend(
            [
                {
                    "name": metric_name,
                    "type": "linear_graph",
                    "x_axis_name": "recalls",
                    "y_axis_name": "precisions",
                    "value": [list(i) for i in zip(raw_data["recalls"], raw_data["precisions"])],
                }
            ]
        )
        return metric_data


class ModelMonitorModelQualityParser(ParserBase):
    """Top level parser for model monitor model quality metric type"""

    def _validate(self, json_data: dict):
        """Implement ParserBase._validate.

        Args:
            json_data (dict): Metric data to be validated.

        Raises:
            ValueError: missing model monitor model quality metrics.
        """
        if len(json_data) == 0:
            raise ValueError("Missing model monitor model quality metrics from the metric data.")

    def _parse(self, json_data: dict):
        """Implement ParserBase._parse.

        Args:
            json_data (dict): Raw metric data.
        """
        result = {"metric_groups": []}
        if "regression_metrics" in json_data:
            result = RegressionParser().run(json_data)
        elif (
            "binary_classification_metrics" in json_data
            or "multiclass_classification_metrics" in json_data
        ):
            result = ClassificationParser().run(json_data)

        return result


EVALUATION_METRIC_PARSERS = {
    EvaluationMetricTypeEnum.MODEL_CARD_METRIC_SCHEMA: DefaultParser(),
    EvaluationMetricTypeEnum.CLARIFY_BIAS: ClarifyBiasParser(),
    EvaluationMetricTypeEnum.CLARIFY_EXPLAINABILITY: ClarifyExplainabilityParser(),
    EvaluationMetricTypeEnum.REGRESSION: RegressionParser(),
    EvaluationMetricTypeEnum.BINARY_CLASSIFICATION: ClassificationParser(),
    EvaluationMetricTypeEnum.MULTICLASS_CLASSIFICATION: ClassificationParser(),
    EvaluationMetricTypeEnum.MODEL_MONITOR_MODEL_QUALITY: ModelMonitorModelQualityParser(),
}
