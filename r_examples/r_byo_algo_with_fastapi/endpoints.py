from typing import Optional, List
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import numpy as np


# Define our expected input types
class Example(BaseModel):
    features: List[List[float]]


# Create a function that we can use to pass our inference function
# to the endpoints during initialization.
def make_endpoints(r_inference_func):
    app = FastAPI()

    @app.get("/ping")
    async def check_health():
        return {"Status": "Alive"}

    @app.post("/invocations")
    async def read_item(input: Example):
        output = r_inference_func(np.array(input.features))
        return {"output": output}

    return app


# A function we can call from R to launch the FastAPI application
def run_app(app):
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info")

