import logging
import azure.functions as func
import numpy as np
import os
import onnxruntime as ort
import json


app = func.FunctionApp()

def preprocess(input_data_json):
    # convert the JSON data into the tensor input
    return np.array(input_data_json['data']).astype('float32')
    
def run_model(model_path, req_body):
    session = ort.InferenceSession(model_path)
    input_data = preprocess(req_body)
    logging.info(f"Input Data shape is {input_data.shape}.")
    input_name = session.get_inputs()[0].name  # get the id of the first input of the model   
    try:
        result = session.run([], {input_name: input_data})
    except (RuntimeError) as e:
        print("Shape={0} and error={1}".format(input_data.shape, e))
    return result[0] 

def get_model_path():
    d=os.path.dirname(os.path.abspath(__file__))
    return os.path.join(d , './model/mnist-pytorch.onnx')

@app.function_name(name="mnist_classify")
@app.route(route="classify", auth_level=func.AuthLevel.ANONYMOUS)
def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    # Get the img value from the post.
    try:
        req_body = req.get_json()
    except ValueError:
        pass

    if req_body:
        # run model
        result = run_model(get_model_path(), req_body)
        # map output to integer and return result string.
        digits = np.argmax(result, axis=1)
        logging.info(type(digits))
        return func.HttpResponse(json.dumps({"digits": np.array(digits).tolist()}))
    else:
        return func.HttpResponse(
             "This HTTP triggered function successfully.",
             status_code=200
        )