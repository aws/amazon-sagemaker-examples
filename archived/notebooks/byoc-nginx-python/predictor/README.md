# Build Predictor (XGBoost) Model

We demonstrate building a ML application to predict the rings of Abalone.

## Open [predictor.ipynb](./predictor.ipynb)

---

The raw payload is first received by the featurizer container. The raw payload is then transformed (feature-engineering) by the featurizer, and the transformed record (float values) are returned as a csv string by the featurizer.

The transformed record is then passed to the predictor container (XGBoost model). The predictor then converts the transformed record into XGBMatrix format, loads the model, calls `booster.predict(input_data)` and returns the predictions (Rings) in a JSON format.

![Abalone Predictor](../images/byoc-predictor.png)

We use nginx as the reverse proxy, gunicorn as the web server gateway interface and the inference code as python Flask app.

>NOTE: A pre-trained XGBoost model is provided in [models](./models/) directory.
