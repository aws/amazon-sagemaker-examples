### this file finds the tf model version in a direcory by getting rid of the file extension
### input example:
import os
import sys
path_to_model = sys.argv[1]

model = os.path.splitext(path_to_model)[0]

print(model)
