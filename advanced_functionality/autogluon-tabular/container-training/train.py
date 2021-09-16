import argparse
import ast
import glob
import json
import logging
import os
import pickle
import shutil
import subprocess
import sys
import warnings
from collections import Counter
from timeit import default_timer as timer

import boto3
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.DEBUG)
logging.info(subprocess.call("ls -lR /opt/ml/input".split()))


import shap
import smdebug.mxnet as smd
from smdebug.core.writer import FileWriter

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import autogluon as ag
    from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION, SOFTCLASS
    from autogluon.tabular import TabularDataset, TabularPredictor
    from prettytable import PrettyTable

    # print(f'DEBUG AutoGluon version : {ag.__version__}')


# ------------------------------------------------------------ #
# Training methods                                             #
# ------------------------------------------------------------ #


def du(path):
    """disk usage in human readable format (e.g. '2,1GB')"""
    return subprocess.check_output(["du", "-sh", path]).split()[0].decode("utf-8")


def __load_input_data(path: str) -> TabularDataset:
    """
    Load training data as dataframe
    :param path:
    :return: DataFrame
    """
    input_data_files = os.listdir(path)

    try:
        input_dfs = [pd.read_csv(f"{path}/{data_file}") for data_file in input_data_files]

        return TabularDataset(data=pd.concat(input_dfs))
    except:
        print(f"No csv data in {path}!")
        return None


def format_for_print(df):
    table = PrettyTable(list(df.columns))
    for row in df.itertuples():
        table.add_row(row[1:])
    return str(table)


def get_roc_auc(y_test_true, y_test_pred, labels, class_labels_internal, model_output_dir):
    from itertools import cycle

    from sklearn.metrics import auc, roc_curve
    from sklearn.preprocessing import label_binarize

    y_test_true_binalized = label_binarize(y_test_true, classes=labels)

    if len(labels) == 2:
        # binary classification

        true_label_index = class_labels_internal.index(1)

        y_test_pred = y_test_pred.values[:, true_label_index]
        y_test_pred = np.reshape(y_test_pred, (-1, 1))
        labels = labels[true_label_index : true_label_index + 1]
        n_classes = 1
    else:
        # multiclass classification
        n_classes = len(labels)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_true_binalized[:, i], y_test_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_true_binalized.ravel(), y_test_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    sns.set(font_scale=1)
    plt.figure()
    lw = 2
    colors = cycle(["aqua", "darkorange", "cornflowerblue"])

    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=lw,
            label=f"ROC curve for {labels[i]} (area = %0.2f)" % roc_auc[i],
        )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic example")
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig(f"{model_output_dir}/roc_auc_curve.png")


def train(args):
    model_output_dir = f"{args.output_dir}/data"

    is_distributed = len(args.hosts) > 1
    host_rank = args.hosts.index(args.current_host)
    dist_ip_addrs = args.hosts
    dist_ip_addrs.pop(host_rank)

    # Load training and validation data
    print(f"Train files: {os.listdir(args.train)}")
    train_data = __load_input_data(args.train)

    # Extract column info
    target = args.init_args["label"]
    columns = train_data.columns.tolist()
    column_dict = {"columns": columns}
    with open("columns.pkl", "wb") as f:
        pickle.dump(column_dict, f)

    # Train models

    args.init_args["path"] = args.model_dir
    # args.fit_args.pop('label', None)
    predictor = TabularPredictor(**args.init_args).fit(train_data, **args.fit_args)

    # Results summary
    predictor.fit_summary(verbosity=3)
    # model_summary_fname_src = os.path.join(predictor.output_directory, 'SummaryOfModels.html')
    model_summary_fname_src = os.path.join(args.model_dir, "SummaryOfModels.html")
    model_summary_fname_tgt = os.path.join(model_output_dir, "SummaryOfModels.html")

    if os.path.exists(model_summary_fname_src):
        shutil.copy(model_summary_fname_src, model_summary_fname_tgt)

    # ensemble visualization
    G = predictor._trainer.model_graph
    remove = [node for node, degree in dict(G.degree()).items() if degree < 1]
    G.remove_nodes_from(remove)
    A = nx.nx_agraph.to_agraph(G)
    A.graph_attr.update(rankdir="BT")
    A.node_attr.update(fontsize=10)
    for node in A.iternodes():
        node.attr["shape"] = "rectagle"
    A.draw(os.path.join(model_output_dir, "ensemble-model.png"), format="png", prog="dot")

    # Optional test data
    if args.test:
        print(f"Test files: {os.listdir(args.test)}")
        test_data = __load_input_data(args.test)
        # Test data must be labeled for scoring
        if target in test_data:
            # Leaderboard on test data
            print("Running model on test data and getting Leaderboard...")
            leaderboard = predictor.leaderboard(test_data, silent=True)
            print(format_for_print(leaderboard), end="\n\n")
            leaderboard.to_csv(f"{model_output_dir}/leaderboard.csv", index=False)

            # Feature importance on test data
            # Note: Feature importance must be calculated on held-out (test) data.
            # If calculated on training data it will be biased due to overfitting.
            if args.feature_importance:
                print("Feature importance:")
                # Increase rows to print feature importance
                pd.set_option("display.max_rows", 500)
                feature_importance_df = predictor.feature_importance(test_data)

                print(feature_importance_df)
                feature_importance_df.to_csv(
                    f"{model_output_dir}/feature_importance.csv", index=True
                )

            # Classification report and confusion matrix for classification model
            if predictor.problem_type in [BINARY, MULTICLASS]:
                from sklearn.metrics import classification_report, confusion_matrix

                X_test = test_data.drop(target, axis=1)
                y_test_true = test_data[target]
                y_test_pred = predictor.predict(X_test)
                y_test_pred_prob = predictor.predict_proba(X_test, as_multiclass=True)

                report_dict = classification_report(
                    y_test_true, y_test_pred, output_dict=True, labels=predictor.class_labels
                )
                report_dict_df = pd.DataFrame(report_dict).T
                report_dict_df.to_csv(f"{model_output_dir}/classification_report.csv", index=True)

                cm = confusion_matrix(y_test_true, y_test_pred, labels=predictor.class_labels)
                cm_df = pd.DataFrame(cm, predictor.class_labels, predictor.class_labels)
                sns.set(font_scale=1)
                cmap = "coolwarm"
                sns.heatmap(cm_df, annot=True, fmt="d", cmap=cmap)
                plt.title("Confusion Matrix")
                plt.ylabel("true label")
                plt.xlabel("predicted label")
                plt.show()
                plt.savefig(f"{model_output_dir}/confusion_matrix.png")

                get_roc_auc(
                    y_test_true,
                    y_test_pred_prob,
                    predictor.class_labels,
                    predictor.class_labels_internal,
                    model_output_dir,
                )
        else:
            warnings.warn("Skipping eval on test data since label column is not included.")

    # Files summary
    print(f"Model export summary:")
    print(f"/opt/ml/model/: {os.listdir('/opt/ml/model/')}")
    models_contents = os.listdir("/opt/ml/model/models")
    print(f"/opt/ml/model/models: {models_contents}")
    print(f"/opt/ml/model directory size: {du('/opt/ml/model/')}\n")


# ------------------------------------------------------------ #
# Training execution                                           #
# ------------------------------------------------------------ #


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.register("type", "bool", lambda v: v.lower() in ("yes", "true", "t", "1"))

    # Environment parameters
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--output-dir", type=str, default=os.environ["SM_OUTPUT_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    # Arguments to be passed to TabularPredictor()
    parser.add_argument(
        "--init_args",
        type=lambda s: ast.literal_eval(s),
        default="{'label': 'y'}",
        help="https://auto.gluon.ai/stable/_modules/autogluon/tabular/predictor/predictor.html#TabularPredictor",
    )
    # Arguments to be passed to task.fit()
    parser.add_argument(
        "--fit_args",
        type=lambda s: ast.literal_eval(s),
        default="{'presets': ['optimize_for_deployment']}",
        help="https://auto.gluon.ai/stable/_modules/autogluon/tabular/predictor/predictor.html#TabularPredictor",
    )
    # Additional options
    parser.add_argument("--feature_importance", type="bool", default=True)

    return parser.parse_args()


if __name__ == "__main__":
    start = timer()
    args = parse_args()

    # Verify label is included
    if "label" not in args.init_args:
        raise ValueError('"label" is a required parameter of "init_args"!')

    # Convert optional fit call hyperparameters from strings
    if "hyperparameters" in args.fit_args:
        for model_type, options in args.fit_args["hyperparameters"].items():
            assert isinstance(options, dict)
            for k, v in options.items():
                args.fit_args["hyperparameters"][model_type][k] = eval(v)

    # Print SageMaker args
    print("fit_args:")
    for k, v in args.fit_args.items():
        print(f"{k},  type: {type(v)},  value: {v}")

    # Make test data optional
    if os.environ.get("SM_CHANNEL_TESTING"):
        args.test = os.environ["SM_CHANNEL_TESTING"]
    else:
        args.test = None

    train(args)

    # Package inference code with model export
    subprocess.call("mkdir /opt/ml/model/code".split())
    subprocess.call("cp /opt/ml/code/inference.py /opt/ml/model/code/".split())
    subprocess.call("cp columns.pkl /opt/ml/model/code/".split())

    elapsed_time = round(timer() - start, 3)
    print(f"Elapsed time: {elapsed_time} seconds. Training Completed!")
