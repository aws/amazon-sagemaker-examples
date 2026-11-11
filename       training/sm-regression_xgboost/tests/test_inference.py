# SPDX-License-Identifier: Apache-2.0
import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "inference.py"


def load_inference():
    package = ModuleType("sagemaker_xgboost_container")
    encoder = ModuleType("sagemaker_xgboost_container.encoder")
    package.encoder = encoder
    spec = importlib.util.spec_from_file_location("regression_inference", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {package.__name__: package, encoder.__name__: encoder}):
        spec.loader.exec_module(module)
    return module


class CsvOutputTests(unittest.TestCase):
    def setUp(self):
        self.inference = load_inference()

    def test_serializes_every_prediction_and_contribution_row(self):
        predictions = np.array([[1.5, 0.1, 0.2], [2.5, 0.3, 0.4], [3.5, 0.5, 0.6]])
        self.assertEqual(
            self.inference.output_fn(predictions, "text/csv"),
            "1.5,0.1,0.2\n2.5,0.3,0.4\n3.5,0.5,0.6",
        )

    def test_single_row_format_is_unchanged(self):
        self.assertEqual(
            self.inference.output_fn(np.array([[1.5, -0.1, 0.0]]), "text/csv"),
            "1.5,-0.1,0.0",
        )

    def test_predict_to_output_preserves_record_order_and_shap_columns(self):
        model = Mock()
        model.predict.side_effect = [
            np.array([10.0, 20.0]),
            np.array([[0.1, 0.2, 9.7], [0.3, 0.4, 19.3]]),
        ]
        inputs = object()
        predictions = self.inference.predict_fn(inputs, model)
        self.assertEqual(
            self.inference.output_fn(predictions, "text/csv"),
            "10.0,0.1,0.2,9.7\n20.0,0.3,0.4,19.3",
        )
        model.predict.assert_any_call(inputs, pred_contribs=True, validate_features=False)

    def test_unsupported_content_type_still_raises(self):
        with self.assertRaisesRegex(ValueError, "application/json is not supported"):
            self.inference.output_fn(np.array([[1.0]]), "application/json")


if __name__ == "__main__":
    unittest.main()
