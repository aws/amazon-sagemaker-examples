import json
import random


# sample preprocess_handler (to be implemented by customer)
# This is a trivial example, where we simply generate random values
# But customers can read the data from inference_record and trasnform it into
# a flattened json structure
def preprocess_handler(inference_record):
    event_data = inference_record.event_data
    input_data = {}
    output_data = {}

    input_data["feature000"] = random.randint(1, 3)
    input_data["feature001"] = random.uniform(0, 1.6)
    input_data["feature002"] = random.uniform(0, 1.6)

    output_data["prediction0"] = random.uniform(1, 30)

    return {**input_data, **output_data}
