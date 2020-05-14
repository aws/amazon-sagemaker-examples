import os
os.system('pip3 install gluoncv')

import gluoncv
import numpy as np
import mxnet as mx

model = gluoncv.model_zoo.get_model('fcn_resnet50_ade', pretrained=True)

# Convert the model to symbolic format
model.hybridize()

# Build a fake image to run a single prediction
# This is required to initialize the model properly
x = np.zeros([1,3,480,480])
x = mx.nd.array(x)

# Predict the fake image
model.forward(x)

# Export the model
model.export('fcn_resnet50_ade')