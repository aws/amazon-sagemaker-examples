import json
import argparse
import joblib
import os
import sys
import numpy as np
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sentence_transformers import SentenceTransformer
from sklearn.utils import class_weight
from sklearn.metrics import f1_score, accuracy_score


def log_cross_val_auc(clf, X, y, cv_splits, log_prefix):
    cv_auc = cross_val_score(clf, X, y, cv=cv_splits, scoring='roc_auc')
    cv_auc_mean = cv_auc.mean()
    cv_auc_error = cv_auc.std() * 2
    log = "{}_auc_cv: {:.5f} (+/- {:.5f})"
    print(log.format(log_prefix, cv_auc_mean, cv_auc_error))


def log_auc(clf, X, y, log_prefix):
    y_pred_proba = clf.predict_proba(X)
    auc = roc_auc_score(y, y_pred_proba[:, 1])
    log = '{}_auc: {:.5f}'
    print(log.format(log_prefix, auc))


def parse_args(sys_args):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sentence-transformer",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2"
    )    
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100
    )
    parser.add_argument(
        "--criterion",
        type=str,
        default="gini"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=-1
    ) 
    parser.add_argument(
        "--min-impurity-decrease",
        type=float,
        default=0.0
    )     
    parser.add_argument(
        "--ccp-alpha",
        type=float,
        default=0.0
    )  
    parser.add_argument(
        "--bootstrap",
        type=str,
        default="True"
    )      
    parser.add_argument(
        "--min-samples-split",
        type=int,
        default=2
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=1
    )
    parser.add_argument(
        "--balanced-data",
        type=str,
        default="True"
    )      
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ.get("SM_MODEL_DIR")
    )
    parser.add_argument(
        "--data-train",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN"),
    )
    parser.add_argument(
        "--data-validation",
        type=str,
        default=os.environ.get("SM_CHANNEL_VALIDATION")
    )
    parser.add_argument(
        "--numerical-feature-names",
        type=str
    )
    parser.add_argument(
        "--categorical-feature-names",
        type=str
    )
    parser.add_argument(
        "--textual-feature-names",
        type=str
    )
    parser.add_argument(
        "--label-name",
        type=str
    )

    args, _ = parser.parse_known_args(sys_args)
    print(args)
    return args


def load_jsonl(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    data = [json.loads(line) for line in lines]
    return data


def extract_labels(
    data,
    label_name
):
    labels = []
    for sample in data:
        value = sample[label_name]
        labels.append(value)
    labels = np.array(labels).astype('int')
    return labels


def extract_numerical_features(
    sample,
    numerical_feature_names
):
    output = []
    for feature_name in numerical_feature_names:
        value = sample[feature_name]
        if value is None:
            value = np.nan
        output.append(value)
    return output


def extract_categorical_features(
    sample,
    categorical_feature_names
):
    output = []
    for feature_name in categorical_feature_names:
        value = sample[feature_name]
        if value is None:
            value = ""
        output.append(value)
    return output


def extract_textual_features(
    sample,
    textual_feature_names
):
    output = []
    for feature_name in textual_feature_names:
        value = sample[feature_name]
        if value is None:
            value = ""
        output.append(value)
    return output


def extract_features(
    data,
    numerical_feature_names,
    categorical_feature_names,
    textual_feature_names
):
    numerical_features = []
    categorical_features = []
    textual_features = []
    for sample in data:
        num_feat = extract_numerical_features(sample, numerical_feature_names)
        numerical_features.append(num_feat)
        cat_feat = extract_categorical_features(sample, categorical_feature_names)
        categorical_features.append(cat_feat)
        text_feat = extract_textual_features(sample, textual_feature_names)
        textual_features.append(text_feat)
    return numerical_features, categorical_features, textual_features


class BertEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
        self.model.parallel_tokenization = False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        output = []
        for sample in X:
            encodings = self.model.encode(sample)
            output.append(encodings)
        return output


def save_feature_names_sentence_transformer(
    numerical_feature_names,
    categorical_feature_names,
    textual_feature_names,
    sentence_transformer,
    filepath
):
    feature_names = {
        'numerical': numerical_feature_names,
        'categorical': categorical_feature_names,
        'textual': textual_feature_names,
        'sentence_transformer': sentence_transformer,
    }
    with open(filepath, 'w') as f:
        json.dump(feature_names, f)
        
        
def load_feature_names_sentence_transformer(filepath):
    with open(filepath, 'r') as f:
        feature_names = json.load(f)
    numerical_feature_names = feature_names['numerical']
    categorical_feature_names = feature_names['categorical']
    textual_feature_names = feature_names['textual']
    sentence_transformer = feature_names['sentence_transformer']
    return numerical_feature_names, categorical_feature_names, textual_feature_names, sentence_transformer
    
def find_filepath(path):
    jsonl_filepaths = [f for f in Path(path).glob('**/*.jsonl')]
    assert len(jsonl_filepaths) == 1, "Single JSON Lines file expected."
    jsonl_filepath = jsonl_filepaths[0]
    return jsonl_filepath

def concatenate_features(numerical_features, categorical_features, textual_features):
    categorical_features = categorical_features.toarray()
    textual_features = np.array(textual_features)
    textual_features = textual_features.reshape(textual_features.shape[0], -1)
    features = np.concatenate([
        numerical_features,
        categorical_features,
        textual_features
    ], axis=1)    
    return features

def train_fn(args):
    # load data
    print('loading training data')
    train_data = load_jsonl(
        find_filepath(args.data_train)
    )
    print(f'length of training data: {len(train_data)}')

    print('loading test data')
    validation_data = load_jsonl(
        find_filepath(args.data_validation)
    )
    print(f'length of validation data: {len(validation_data)}')
    
    # parse feature names
    print('parsing feature names')
    numerical_feature_names = args.numerical_feature_names.split(',')
    categorical_feature_names = args.categorical_feature_names.split(',')
    textual_feature_names = args.textual_feature_names.split(',')
    
    print('saving feature names')
    save_feature_names_sentence_transformer(
        numerical_feature_names,
        categorical_feature_names,
        textual_feature_names,
        args.sentence_transformer,
        Path(args.model_dir, "feature_names.json")
    )
    
    # extract label
    print('extracting label')
    train_labels = extract_labels(train_data, args.label_name)
    print(f'length of training labels: {len(train_labels)}')
    validation_labels = extract_labels(validation_data, args.label_name)
    print(f'length of validation labels: {len(validation_labels)}') 
    
    
    if args.balanced_data:
        print('computing class weights')
        train_val_labels = np.concatenate((train_labels, validation_labels), axis=0)
        class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(train_val_labels), y=train_val_labels)
        class_weight_dict = {}
        for idx, weight in enumerate(class_weights):
            class_weight_dict[idx] = weight
    else:
        class_weight_dict = None
    

    # extract features
    print('extracting features for training and validation data')
    train_numerical_features, train_categorical_features, train_textual_features = extract_features(
        train_data,
        numerical_feature_names,
        categorical_feature_names,
        textual_feature_names
    )
    
    validation_numerical_features, validation_categorical_features, validation_textual_features = extract_features(
        validation_data,
        numerical_feature_names,
        categorical_feature_names,
        textual_feature_names
    )

    # define preprocessors
    print('defining preprocessors')
    numerical_transformer = SimpleImputer(missing_values=np.nan, strategy='mean', add_indicator=True)
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    textual_transformer = BertEncoder(model_name=args.sentence_transformer)

    # fit and save preprocessors
    print('fitting numerical_transformer')
    numerical_transformer.fit(train_numerical_features + validation_numerical_features)
    print('saving categorical_transformer')
    joblib.dump(numerical_transformer, Path(args.model_dir, "numerical_transformer.joblib"))
    print('fitting categorical_transformer')
    categorical_transformer.fit(train_categorical_features + validation_categorical_features)
    print('saving categorical_transformer')
    joblib.dump(categorical_transformer, Path(args.model_dir, "categorical_transformer.joblib"))

    # transform features
    print('transforming numerical_features for training and validataion data')
    train_numerical_features = numerical_transformer.transform(train_numerical_features)
    validation_numerical_features = numerical_transformer.transform(validation_numerical_features)
    print('transforming categorical_features for training and validataion data')
    train_categorical_features = categorical_transformer.transform(train_categorical_features)
    validation_categorical_features = categorical_transformer.transform(validation_categorical_features)
    print('transforming textual_features for training and validataion data')
    train_textual_features = textual_transformer.transform(train_textual_features)
    validation_textual_features = textual_transformer.transform(validation_textual_features)

    # concat features
    print('concatenating features')
    train_features = concatenate_features(train_numerical_features, train_categorical_features, train_textual_features)
    validation_features = concatenate_features(validation_numerical_features, validation_categorical_features, validation_textual_features)
    
    if args.max_depth == -1:
        max_depth = None
    else:
        max_depth = args.max_depth
        
    if args.bootstrap == "True":
        bootstrap = True
    else:
        bootstrap = False
    # define model
    print('instantiating model')
    classifier = RandomForestClassifier(
        n_estimators=args.n_estimators,
        criterion=args.criterion,
        max_depth=max_depth,
        min_impurity_decrease=args.min_impurity_decrease,
        ccp_alpha=args.ccp_alpha,
        bootstrap=bootstrap,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        class_weight=class_weight_dict,
    )

    # fit and save model
    print('fitting model')
    classifier = classifier.fit(train_features, train_labels)
    
    print('evaluating the model on the validation data')
    prediction_prob = classifier.predict_proba(validation_features).tolist()
    prediction_prob = np.array(prediction_prob)
    prediction_labels = np.argmax(prediction_prob, axis=1)
    
    f1_score_val = f1_score(validation_labels, prediction_labels)
    accuracy_val = accuracy_score(validation_labels, prediction_labels)
    roc_auc_val = roc_auc_score(validation_labels, prediction_prob[:, 1])
    print(f'f1 score on validation data: {f1_score_val}')
    print(f'accuracy score on validation data: {accuracy_val}')
    print(f'roc auc score on validation data: {roc_auc_val}')
    
    print('saving model')
    joblib.dump(classifier, Path(args.model_dir, "classifier.joblib"))


### DEPLOYMENT FUNCTIONS
def model_fn(model_dir):
    print('loading feature_names')
    numerical_feature_names, categorical_feature_names, textual_feature_names, sentence_transformer = load_feature_names_sentence_transformer(Path(model_dir, "feature_names.json"))
    print('loading numerical_transformer')
    numerical_transformer = joblib.load(Path(model_dir, "numerical_transformer.joblib"))
    print('loading categorical_transformer')
    categorical_transformer = joblib.load(Path(model_dir, "categorical_transformer.joblib"))
    print('loading textual_transformer')
    textual_transformer = BertEncoder(model_name=sentence_transformer)
    classifier = joblib.load(Path(model_dir, "classifier.joblib"))
    model_assets = {
        'numerical_feature_names': numerical_feature_names,
        'numerical_transformer': numerical_transformer,
        'categorical_feature_names': categorical_feature_names,
        'categorical_transformer': categorical_transformer,
        'textual_feature_names': textual_feature_names,
        'textual_transformer': textual_transformer,
        'classifier': classifier
    }
    return model_assets


def input_fn(request_body_str, request_content_type):
    assert (
        request_content_type == "application/json"
    ), "content_type must be 'application/json'"
    request_body = json.loads(request_body_str)
    return request_body


def predict_fn(request, model_assets):
    print('extracting features')
    numerical_features, categorical_features, textual_features = extract_features(
        request,
        model_assets['numerical_feature_names'],
        model_assets['categorical_feature_names'],
        model_assets['textual_feature_names']
    )
    
    print('transforming numerical_features')
    numerical_features = model_assets['numerical_transformer'].transform(numerical_features)
    print('transforming categorical_features')
    categorical_features = model_assets['categorical_transformer'].transform(categorical_features)
    print('transforming textual_features')
    textual_features = model_assets['textual_transformer'].transform(textual_features)
    
    # concat features
    print('concatenating features')
    categorical_features = categorical_features.toarray()
    textual_features = np.array(textual_features)
    textual_features = textual_features.reshape(textual_features.shape[0], -1)
    features = np.concatenate([
        numerical_features,
        categorical_features,
        textual_features
    ], axis=1)
    
    print('predicting using model')
    prediction = model_assets['classifier'].predict_proba(features)
    probabilities = prediction.tolist()
    output = {
        'probability': probabilities
    }
    return output


def output_fn(prediction, response_content_type):
    assert (
        response_content_type == "application/json"
    ), "accept must be 'application/json'"
    response_body_str = json.dumps(prediction)
    return response_body_str


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    train_fn(args)
