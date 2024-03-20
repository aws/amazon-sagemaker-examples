# PyTorch MNIST Example

The example shows how to use @remote decorator and RemoteExecutor in Python scripts. 

Directory structure

```
./
  config.yaml
  requirements.txt # list of Python packages to install
  load_data.py # Python function to load MNIST dataset
  model.py # CNN architecture
  train.py # main training function where @remote is applied. 
```

Note that the main script where @remote is applied must be at the root of the workspace.

To execute the training, run
```
python ./train.py
```
