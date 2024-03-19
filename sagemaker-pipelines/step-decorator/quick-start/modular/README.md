This example shows how to structure a Python workspace in modular way for building a SageMaker Pipeline with 
data preprocessing, training and evaluation steps.

Notes:
- This example can only run on either Python 3.8 or Python 3.10. 
Otherwise, you will get an error message prompting you to provide an image_uri when defining a step.
- Please update your configurations, e.g. RoleArn, in the `config.yaml` if needed.
- Please run the commands below under the `modular` folder.

## Install the beta SageMaker Python SDK

```bash
pip install -r requirements.txt
```

## Run the example

```bash
python pipeline.py
```