import argparse, os
import boto3
import sagemaker
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping


#ANN Model
def createModel():
    model = Sequential()
    model.add(Dense(13, input_shape=(13,), activation='relu'))
    model.add(Dense(28, activation='relu'))
    model.add(Dense(13, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

if __name__ == '__main__':
    
    #Passing in environment variables and hyperparameters for our training script
    parser = argparse.ArgumentParser()
    
    #Can have other hyper-params such as batch-size, which we are not defining in this case
    parser.add_argument('--epochs', type=int, default=30)
    
    #sm_model_dir: model artifacts stored here after training
    #training directory has the data for the model
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    
    #Read hyperparams and data
    args, _ = parser.parse_known_args()
    epochs     = args.epochs
    
    #model artifacts and data
    training_dir   = args.train
    
    #Data Reading & Preprocessing
    print("reading")
    df = pd.read_csv(training_dir + '/train.csv',sep=',')
    print("read")
    X = df.drop('TARGET', axis = 1)
    y = df['TARGET']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #minimum epochs 15
    model = createModel()
    monitor_val_acc = EarlyStopping(monitor = 'val_loss', patience=15)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), callbacks=[monitor_val_acc], epochs=epochs)

    print("Saving model")
    model.save(os.path.join(args.sm_model_dir, '00000001'))