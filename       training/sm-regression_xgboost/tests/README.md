# Offline inference regression tests

From the repository root, with Python 3.11–3.13:

```sh
python -m pip install 'numpy>=1.26,<3' 'black==26.10.1'
python -m unittest discover -s '      training/sm-regression_xgboost/tests' -v
python -m black -l 100 --check '      training/sm-regression_xgboost/inference.py' '      training/sm-regression_xgboost/tests'
```

CSV responses contain one record per prediction row, in input order. Each record
contains the predicted value followed by its SHAP contributions. Single-record
responses retain their original formatting, without a trailing newline.

Tests use real NumPy arrays and a mocked model/encoder import boundary. They do
not install the SageMaker XGBoost container, launch jobs, deploy endpoints, or
validate actual model scores. No notebook cells are executed or modified.
