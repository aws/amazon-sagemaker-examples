import argparse
import joblib
import os
import glob
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report

# inference function for model loading
def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf

if __name__ == "__main__":

    print("extracting arguments")
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    # to simplify the demo we don't use all sklearn RandomForest hyperparameters
    parser.add_argument("--learningrate", type=float, default=0.1)
    parser.add_argument("--maxiter", type=int, default=100)

    # Data, model, and output directories
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--validation", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION"))

    args, _ = parser.parse_known_args()
    
    class_list = ['Benign','Bot','DoS attacks-GoldenEye','DoS attacks-Slowloris','DDoS attacks-LOIC-HTTP','Infilteration','DDOS attack-LOIC-UDP','DDOS attack-HOIC','Brute Force-Web','Brute Force-XSS','SQL Injection','DoS attacks-SlowHTTPTest','DoS attacks-Hulk','FTP-BruteForce','SSH-Bruteforce']
    
    # read all data from training folder
    train_csv_files = glob.glob(args.train + "/*.csv")
    print(train_csv_files)
    df_list = (pd.read_csv(file) for file in train_csv_files)
    train_df   = pd.concat(df_list, ignore_index=True)
    #train_df.dropna(inplace=True)

    # read all data from validation file
    val_csv_files = glob.glob(args.validation + "/*.csv")
    print(val_csv_files)
    df_list = (pd.read_csv(file) for file in val_csv_files)
    val_df   = pd.concat(df_list, ignore_index=True)
    #val_df.dropna(inplace=True)
    
    # build training and testing dataset
    print("building training and testing datasets")
    X_train = train_df[train_df.columns[1:]]
    X_val = val_df[val_df.columns[1:]]
    y_train = train_df[train_df.columns[0]]
    y_val = val_df[val_df.columns[0]]
    print(X_train.shape)
    print(X_val.shape)
    print(y_train.shape)
    print(y_val.shape)
    
    # train
    print("training model")
    model = HistGradientBoostingClassifier(learning_rate=args.learningrate,max_iter=args.maxiter,verbose=2,
    )

    model.fit(X_train.values, y_train.values)

    # print accuracy
    print("validating model")
    y_pred = model.predict(X_val.values)
    acc = accuracy_score(y_val, y_pred)
    #auc = roc_auc_score(y_val, y_pred,multi_class='ovo')
    wf1 = f1_score(y_val,y_pred,average='weighted')
    print(f"Accuracy is: {acc}")
    #print(f"Area under the curve is: {auc}")
    print(f"Weighted F1 Score is: {wf1}")
    print()
    print("Classification Report")
    print(classification_report(y_val,y_pred,target_names = class_list))

    # persist model
    path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, path)
    print("model persisted at " + path)