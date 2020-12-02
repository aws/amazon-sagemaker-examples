# Test the Entry Points
To test the entry point `train.py` in your local python environment before 
running on SageMaker:
```
python test_train.py
```

To test the entry point `inference.py`:
```
python test_inference.py
```

Testing the entry points in your local python development environment makes 
debugging easier and faster. Ensure that your local environement has 
all the dependencies required by `train.py`, `test_train.py`, `inference.py` and 
`test_inference.py`. 

## Test Script for `train.py`
`test_train.py` simulates a SageMaker deep learning training container 
environment by creating environment variables required by `train.py` in 
the `Env` class. It downloads the MNIST dataset from a public S3 bucket: 
`sagemaker-sample-files` and save it in a temporary directory `/tmp/data`.
Then, it uses this temporary directory as the training and testing channels
for `train.py`. It does so by setting `SM_CHANNEL_TRAINING` and `SM_CHANNEL_TESTING` to `/tmp/data`. 

`test_train.py` triggers model training by calling the `train` function in 
`train.py` with the default arguments set by `parse_args`. It sets 
`SM_MODEL_DIR` to `/tmp/model`, so that after the training is complete, the
model artifact is saved in `/tmp/model`.

## Test Script for `inference.py`
`test_inference.py` is used to debugging `model_fn` and `transform_fn` implemented
in `inference.py`. The main `test` function simulates one endpoint invocation 
via the following steps:
1. Load the model artifact from `SM_MODEL_DIR` set by `test_train.py` by calling
`model_fn`. This step simulates how the inference container loads the model when
executed. 
2. Create dummy data and serialize it to a json string. This step simulates 
how `sagemaker.predictor.Predictor` class preprocess the input data. 
3. Call `transform_fn` function with the loaded model from step 1 and json string
from step 2. This step simulates how inference container processes incoming 
requests. 

