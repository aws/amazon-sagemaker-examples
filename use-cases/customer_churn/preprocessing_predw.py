import argparse
import glob
import json
import os
import time
import warnings

import pandas as pd
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action="ignore", category=DataConversionWarning)
start_time = time.time()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processing-output-filename")

    args, _ = parser.parse_known_args()
    print("Received arguments {}".format(args))

    input_jsons = glob.glob("/opt/ml/processing/input/data/**/*.json", recursive=True)

    df_all = pd.DataFrame()
    for name in input_jsons:
        print("\nStarting file: {}".format(name))
        df = pd.read_json(name, lines=True)
        df_all = df_all.append(df)

    output_filename = args.processing_output_filename
    final_features_output_path = os.path.join("/opt/ml/processing/output", output_filename)
    print("Saving processed data to {}".format(final_features_output_path))
    df_all.to_csv(final_features_output_path, header=True, index=False)
