import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Define the input and output paths
input_path = '/opt/ml/processing/input/feature-selection-query-id.csv'
train_output_path = '/opt/ml/processing/output/train/train.csv'
val_output_path = '/opt/ml/processing/output/validation/val.csv'
test_output_path = '/opt/ml/processing/output/test/test.csv'

# Read the input data
df = pd.read_csv(input_path, header=None)

# Split the data into training, validation, and test sets
train, temp = train_test_split(df, test_size=0.3, random_state=42)
val, test = train_test_split(temp, test_size=0.5, random_state=42)

# Save the splits to the output paths
os.makedirs(os.path.dirname(train_output_path), exist_ok=True)
train.to_csv(train_output_path, index=False)

os.makedirs(os.path.dirname(val_output_path), exist_ok=True)
val.to_csv(val_output_path, index=False)

os.makedirs(os.path.dirname(test_output_path), exist_ok=True)
test.to_csv(test_output_path, index=False)

# Print the sizes of the splits
print(f"Training set: {len(train)} samples")
print(f"Validation set: {len(val)} samples")
print(f"Test set: {len(test)} samples")
