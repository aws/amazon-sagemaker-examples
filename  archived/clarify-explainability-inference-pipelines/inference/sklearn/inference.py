from __future__ import print_function


from io import StringIO
import os


import pandas as pd


from sklearn.externals import joblib


feature_columns_names = [
    "status",
    "duration",
    "credit_history",
    "purpose",
    "amount",
    "savings",
    "employment_duration",
    "installment_rate",
    "personal_status_sex",
    "other_debtors",
    "present_residence",
    "property",
    "age",
    "other_installment_plans",
    "housing",
    "number_credits",
    "job",
    "people_liable",
    "telephone",
    "foreign_worker",
]


def input_fn(input_data, content_type):

    if content_type == "text/csv":
        df = pd.read_csv(StringIO(input_data), header=None, index_col=False, sep=",")

        first_row = df.iloc[0:1].values[0].tolist()

        if len(df.columns) == len(feature_columns_names):
            print("column length is correct")

            if set(first_row) == set(feature_columns_names):
                print("the row contains header, remove the row")
                df = df.iloc[1:]
                df.reset_index(drop=True, inplace=True)

            df.columns = feature_columns_names

        return df
    else:
        raise ValueError("{} not supported by script!".format(content_type))


def predict_fn(input_data, model):
    input_data.head(1)
    features = model.transform(input_data)
    print("successful sklearn inference", features)
    return features


def model_fn(model_dir):
    preprocessor = joblib.load(os.path.join(model_dir, "model.joblib"))
    return preprocessor
