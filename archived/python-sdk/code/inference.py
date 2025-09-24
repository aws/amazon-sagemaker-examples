import os
import joblib

def predict_fn(input_object, model):
    #########################################
    # Do your custom preprocessing logic here
    #########################################
    predictions = model.predict(input_object)
    return predictions


def model_fn(model_dir):
    print("loading model.joblib from: {}".format(model_dir))
    loaded_model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return loaded_model
