from sklearn.model_selection import train_test_split
import argparse
import os
from io import StringIO
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="parsing all commandline arguments to the processing job"
    )
    
    # parser.add_argument("--s3_input_bucket", type=str, help="s3 bucket containing input data")
    # parser.add_argument("--s3_input_key_prefix", type=str, help="s3 input key prefix")
    # parser.add_argument("--s3_output_bucket", type=str, help="s3 output bucket")
    # parser.add_argument("--s3_output)_key_prefix", type=str, help="s3 output key prefix")
    print("starting near.....")
    processing_root_path = "/opt/ml/processing/"
    input_data_prefix = "input"
    train_output_prefix = "output/train/"
    validation_output_prefix = "output/validation/"
    test_output_prefix = "output/test/"
    
    file_path = processing_root_path + input_data_prefix+"/churn.txt"
    df = pd.read_csv(file_path)
    print(df)
    # Phone number is unique - will not add value to classifier
    df = df.drop("Phone", axis=1)

    # Cast Area Code to non-numeric
    df["Area Code"] = df["Area Code"].astype(object)

    # Remove one feature from highly corelated pairs
    df = df.drop(["Day Charge", "Eve Charge", "Night Charge", "Intl Charge"], axis=1)

    # One-hot encode catagorical features into numeric features
    model_data = pd.get_dummies(df)
    model_data = pd.concat(
        [
            model_data["Churn?_True."],
            model_data.drop(["Churn?_False.", "Churn?_True."], axis=1),
        ],
        axis=1,
    )
    model_data = model_data.astype(float)

    # Split data into train and validation datasets
    train_data, validation_data = train_test_split(
        model_data, test_size=0.33, random_state=42
    )

    # Further split the validation dataset into test and validation datasets.
    validation_data, test_data = train_test_split(
        validation_data, test_size=0.33, random_state=42
    )

    # Remove and store the target column for the test data
    test_target_column = test_data["Churn?_True."]
    test_data.drop(["Churn?_True."], axis=1, inplace=True)
    print("ending near.....")
    # Store all datasets locally
    train_data.to_csv(processing_root_path+train_output_prefix+"train.csv", header=False, index=False)
    validation_data.to_csv(processing_root_path+validation_output_prefix+"validation.csv", header=False, index=False)
    test_data.to_csv(processing_root_path+test_output_prefix+"test.csv", header=False, index=False)
    
if __name__ == "__main__":
    main()