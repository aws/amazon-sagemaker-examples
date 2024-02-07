from __future__ import print_function

import io
import os
import pickle

import pandas as pd
from fastapi import FastAPI, Response, Request

import multiprocessing
cpu_count = multiprocessing.cpu_count()

prefix = "/opt/ml/"
model_path = os.path.join(prefix, "model")

class ScoringService(object):
    model = None  # Where we keep the model when it's loaded

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model is None:
            with open(os.path.join(model_path, "decision-tree-model.pkl"), "rb") as inp:
                cls.model = pickle.load(inp)
        return cls.model

    @classmethod
    def predict(cls, input):
        """For the input, do the predictions and return them.

        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
        clf = cls.get_model()
        return clf.predict(input)


app = FastAPI()


@app.get("/ping")
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ScoringService.get_model() is not None  # You can insert a health check here

    status = 200 if health else 404
    return Response(content="\n",status_code=status,media_type="application/json")

@app.post("/invocations")
async def transformation(request: Request):
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None

    # Convert from CSV to pandas
    content_type = request.headers.get("content-type", None)
    if content_type == "text/csv":
        content = await request.body()
        with io.BytesIO(content) as s:
            data = pd.read_csv(s, header=None)
    else:
        return Response(
            content="This predictor only supports csv data", 
            status_code=415,
            media_type="text/plain"
        )
    
    print("Invoked with {} records".format(data.shape[0]))

    # Do the prediction
    predictions = ScoringService.predict(data)

    # Convert from numpy back to CSV
    out = io.StringIO()
    pd.DataFrame({"results": predictions}).to_csv(out, header=False, index=False)
    result = out.getvalue()

    return Response(
        content=result,
        status_code=200,
        media_type="text/csv"
    )

if __name__=="__main__":
    import uvicorn
    uvicorn.run("predictor:app",host='0.0.0.0', port=8080, workers=cpu_count)

