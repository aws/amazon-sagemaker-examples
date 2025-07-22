# Bird Object Detection Sample Notebook

The Amazon SageMaker notebook is `object_detection_birds.ipynb`.  In addition, we have provided files in the `tools` folder.

1. `patch_ssd.sh` - this is a bash script for patching the model artifacts to be suitable for running on AWS DeepLens.
2. `birdsOnEdge.py` - this is a Python script that can be deployed as AWS Lambda inference function on AWS DeepLens.  It depends on the patched model artifacts and performs the necessary additional step of calling `mo.optimize` before loading the model.
3. `im2rec.py` - this is a copy of a Python script from Apache MXNet that is used by the notebook to create RecordIO files of the bird images.
