import base64
import hashlib
import json
from io import BytesIO, StringIO

import pandas as pd
from PIL import Image

from autogluon.multimodal import MultiModalPredictor


def _decode_and_save_base85_image(image_b85):
    """saves Base65 encoded image to disk and returns unique name"""

    image_bytes = base64.b85decode(image_b85)
    image_hash = hashlib.md5(image_bytes).hexdigest()
    image_name = f"image_{image_hash}.png"
    image = Image.open(BytesIO(image_bytes))
    image.save(image_name)

    return image_name


def model_fn(model_dir):
    """loads model from previously saved artifact"""

    model = MultiModalPredictor.load(model_dir)
    globals()["column_names"] = [
        column_name
        for column_name in model._column_types.keys()
        if column_name != model._label_column
    ]

    return model


def transform_fn(model, request_body, input_content_type, output_content_type="application/json"):

    if input_content_type != "text/csv":
        raise Exception(f"{input_content_type} content type not supported.")

    buf = StringIO(request_body)
    data = pd.read_csv(buf, header=None)

    if len(data.columns) != len(column_names):
        raise Exception(
            f"Invalid data format. Input data has {len(data.columns)} while the model expects {len(column_names)}."
        )

    data.columns = column_names

    for column_name, column_type in model._column_types.items():
        if column_type in ("image_path", "image"):
            data[column_name] = [
                _decode_and_save_base85_image(encoded_image) for encoded_image in data[column_name]
            ]

    prediction_proba = model.predict_proba(data)

    return json.dumps(prediction_proba.to_numpy().tolist()), output_content_type
