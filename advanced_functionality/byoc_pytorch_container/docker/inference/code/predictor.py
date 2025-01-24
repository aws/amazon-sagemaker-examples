# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import io
import os
import flask
import pandas as pd


import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

prefix = "/opt/ml/"
model_path = os.path.join(prefix, "model")

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.

class MultipleRegression(nn.Module):
    def __init__(self, num_features):
        super(MultipleRegression, self).__init__()
        
        self.layer_1 = nn.Linear(num_features, 16)
        self.layer_2 = nn.Linear(16, 32)
        self.layer_3 = nn.Linear(32, 16)
        self.layer_out = nn.Linear(16, 1)
        
        self.relu = nn.ReLU()
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.relu(self.layer_2(x))
        x = self.relu(self.layer_3(x))
        x = self.layer_out(x)
        return (x)

    def predict(self, test_inputs):
        x = self.relu(self.layer_1(test_inputs))
        x = self.relu(self.layer_2(x))
        x = self.relu(self.layer_3(x))
        x = self.layer_out(x)
        return (x)
    
class RegressionDataset(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__ (self):
        return len(self.X_data)
    
class ModelService(object):
    model = None  # Where we keep the model when it's loaded

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""

        
        submodel_file_path = '/opt/ml/model/model_weights.pth'
        cls.model = MultipleRegression(10) 
        cls.model.load_state_dict(torch.load(submodel_file_path))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        return cls.model.to(device)


    @classmethod
    def predict(cls, input):
        """For the input, do the predictions and return them.

        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
        clf = cls.get_model()
        return clf.predict(input)
    
# The flask app for serving predictions
app = flask.Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    root_path = "/opt/ml"

    for r, d, f in os.walk(root_path):
        for file in f:
            print(os.path.join(r, file))
            if file == 'model.tar.gz':
                model_tar_file = os.path.join(r, file)
                print('print the model file path:',model_tar_file)


    health = ModelService.get_model() is not None  # You can insert a health check here

    print('health:',health)
    status = 200 if health else 404
    return flask.Response(response="\n", status=status, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None

    # Convert from CSV to pandas
    if flask.request.content_type == "text/csv":
        data = flask.request.data.decode("utf-8")
        s = io.StringIO(data)
        data = pd.read_csv(s, header=None)
        X_inf = data.values
        
        X_inf = data.iloc[: , :-1].values
        y_inf = data.iloc[: , -1].values

        inf_dataset = RegressionDataset(torch.from_numpy(X_inf).float(), torch.from_numpy(y_inf).float())
        inf_loader = DataLoader(dataset=inf_dataset, batch_size=1000)
        
    else:
        return flask.Response(
            response="This predictor only supports CSV data", status=415, mimetype="text/plain"
        )

    print("Invoked with {} records".format(data.shape[0]))

    # Do the prediction
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for X_inf_batch, y_inf_batch in inf_loader:
        X_inf_batch, y_inf_batch = X_inf_batch.to(device), y_inf_batch.to(device)
        predictions = ModelService.predict(X_inf_batch)
        print(predictions)
        

    # Convert from numpy back to CSV
    out = io.StringIO()
    pd.DataFrame({"results": torch.transpose(predictions, 0, 1).tolist()[0]}).to_csv(out)
    result = out.getvalue()


    return flask.Response(response=result, status=200, mimetype="text/csv")