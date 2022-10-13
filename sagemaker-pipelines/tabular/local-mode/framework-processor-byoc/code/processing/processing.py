import argparse
import csv
import logging
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
from sklearn.model_selection import train_test_split
import traceback

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

BASE_PATH = os.path.join("/", "opt", "ml")
PROCESSING_PATH = os.path.join(BASE_PATH, "processing")
PROCESSING_PATH_INPUT = os.path.join(PROCESSING_PATH, "input")
PROCESSING_PATH_OUTPUT = os.path.join(PROCESSING_PATH, "output")

def extract_data(file_path, percentage=100):
    try:
        files = [f for f in listdir(file_path) if isfile(join(file_path, f)) and f.endswith(".csv")]
        LOGGER.info("{}".format(files))

        frames = []

        for file in files:
            df = pd.read_csv(
                os.path.join(file_path, file),
                sep=",",
                quotechar='"',
                quoting=csv.QUOTE_ALL,
                escapechar='\\',
                encoding='utf-8',
                error_bad_lines=False
            )

            df = df.head(int(len(df) * (percentage / 100)))

            frames.append(df)

        df = pd.concat(frames)

        return df
    except Exception as e:
        stacktrace = traceback.format_exc()
        LOGGER.error("{}".format(stacktrace))

        raise e

def load_data(df, file_path, file_name):
    try:
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        path = os.path.join(file_path, file_name + ".csv")

        LOGGER.info("Saving file in {}".format(path))

        df.to_csv(
            path,
            index=False,
            header=True,
            quoting=csv.QUOTE_ALL,
            encoding="utf-8",
            escapechar="\\",
            sep=","
        )
    except Exception as e:
        stacktrace = traceback.format_exc()
        LOGGER.error("{}".format(stacktrace))

        raise e

def transform_data(df):
    try:
        df = df[df['Is Fraud?'].notna()]

        df.insert(0, 'ID', range(1, len(df) + 1))

        df["Errors?"].fillna('', inplace=True)
        df['Errors?'] = df['Errors?'].map(lambda x: x.strip())
        df["Errors?"] = df["Errors?"].map({
            "Insufficient Balance": 0,
            "Technical Glitch": 1,
            "Bad PIN": 2,
            "Bad Expiration": 3,
            "Bad Card Number": 4,
            "Bad CVV": 5,
            "Bad PIN,Insufficient Balance": 6,
            "Bad PIN,Technical Glitch": 7,
            "": 8
        })

        df["Use Chip"].fillna('', inplace=True)
        df['Use Chip'] = df['Use Chip'].map(lambda x: x.strip())
        df["Use Chip"] = df["Use Chip"].map({
            "Swipe Transaction": 0,
            "Chip Transaction": 1,
            "Online Transaction": 2
        })

        df['Is Fraud?'] = df['Is Fraud?'].map(lambda x: x.replace("'", ""))
        df['Is Fraud?'] = df['Is Fraud?'].map(lambda x: x.strip())
        df['Is Fraud?'] = df['Is Fraud?'].replace('', np.nan)
        df['Is Fraud?'] = df['Is Fraud?'].replace(' ', np.nan)

        df["Is Fraud?"] = df["Is Fraud?"].map({"No": 0, "Yes": 1})

        df = df.rename(
            columns={'Card': 'card', 'MCC': 'mcc', "Errors?": "errors", "Use Chip": "use_chip", "Is Fraud?": "labels"})

        df = df[["card", "mcc", "errors", "use_chip", "labels"]]

        return df

    except Exception as e:
        stacktrace = traceback.format_exc()
        LOGGER.error("{}".format(stacktrace))

        raise e

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-percentage", type=int, required=False, default=100)
    args = parser.parse_args()

    LOGGER.info("Arguments: {}".format(args))

    df = extract_data(PROCESSING_PATH_INPUT, args.dataset_percentage)

    df = transform_data(df)

    data_train, data_test = train_test_split(df, test_size=0.2, shuffle=True)

    load_data(data_train, os.path.join(PROCESSING_PATH_OUTPUT, "train"), "train")
    load_data(data_test, os.path.join(PROCESSING_PATH_OUTPUT, "test"), "test")
