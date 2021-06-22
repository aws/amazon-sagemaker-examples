
from joblib import dump, load
import pandas as pd, numpy as np, os, argparse

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Argument parser
def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type=str, default='/opt/ml/processing/input/')
    parser.add_argument('--filename', type=str, default='data.csv')
    parser.add_argument('--outputpath', type=str, default='/opt/ml/processing/output/')
    # Parse the arguments
    return parser.parse_known_args()

# Main Training Loop
if __name__=="__main__":
    # Process arguments
    args, _ = _parse_args()
    # Load the dataset
    df = pd.read_csv(os.path.join(args.filepath, args.filename))
    X,y = df.drop('y', axis=1), df.y
    # Define the pipeline and train it
    pipe = Pipeline([('scaler', StandardScaler())])
    transformed = pipe.fit_transform(X)
    # Generate the output files - train and test
    output = pd.concat([pd.DataFrame(transformed), y], axis=1)
    train, test = train_test_split(output, random_state=42)
    train.to_csv(os.path.join(args.outputpath, 'train/train.csv'), index=False)
    test.to_csv(os.path.join(args.outputpath, 'test/test.csv'), index=False)
    # Store the pipeline
    dump(pipe, os.path.join(args.outputpath, 'pipeline/preproc-pipeline.joblib'))
