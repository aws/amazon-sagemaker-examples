from argparse import ArgumentParser, Namespace
import csv
import emoji
import json
import logging
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import pickle
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from time import gmtime, strftime
import torch
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"


class MulticlassClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MulticlassClassifier, self).__init__()

        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        self.relu2 = torch.nn.ReLU()
        self.dr1 = torch.nn.Dropout(0.2)
        self.fc3 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dr1(out)
        out = self.fc3(out)

        return out


def __read_params():
    try:
        parser = ArgumentParser()

        parser.add_argument("--epochs", type=int, default=25)
        parser.add_argument("--learning_rate", type=float, default=0.001)
        parser.add_argument("--batch_size", type=int, default=100)
        parser.add_argument("--dataset_percentage", type=str, default=100)
        parser.add_argument(
            "--output-data-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR")
        )
        parser.add_argument(
            "--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN")
        )
        parser.add_argument(
            "--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR")
        )

        args = parser.parse_args()

        if len(vars(args)) == 0:
            with open(
                os.path.join(
                    "/", "opt", "ml", "input", "config", "hyperparameters.json"
                ),
                "r",
            ) as f:
                training_params = json.load(f)

            args = Namespace(**training_params)

        return args
    except Exception as e:
        stacktrace = traceback.format_exc()
        logger.error("{}".format(stacktrace))

        raise e


def clean_text(text):
    text = text.lower()

    text = text.lstrip()
    text = text.rstrip()

    text = re.sub("\[.*?\]", "", text)
    text = re.sub("https?://\S+|www\.\S+", "", text)
    text = re.sub("\n", "", text)
    text = " ".join(filter(lambda x: x[0] != "@", text.split()))

    text = emoji.replace_emoji(text, "")

    text = text.replace("u'", "'")

    text = text.encode("ascii", "ignore")
    text = text.decode()

    word_list = text.split(" ")

    for word in word_list:
        if isinstance(word, bytes):
            word = word.decode("utf-8")

    text = " ".join(word_list)

    if not any(c.isalpha() for c in text):
        return ""
    else:
        return text


def extract_data(file_path, percentage=100):
    try:
        files = [
            f
            for f in listdir(file_path)
            if isfile(join(file_path, f)) and f.endswith(".csv")
        ]
        logger.info("{}".format(files))

        frames = []

        for file in files:
            df = pd.read_csv(
                os.path.join(file_path, file),
                sep=",",
                quotechar='"',
                quoting=csv.QUOTE_ALL,
                escapechar="\\",
                encoding="utf-8",
                on_bad_lines="skip",
            )

            df = df.head(int(len(df) * (percentage / 100)))

            frames.append(df)

        df = pd.concat(frames)

        return df
    except Exception as e:
        stacktrace = traceback.format_exc()
        logger.error("{}".format(stacktrace))

        raise e


def load_data(df, file_path, file_name):
    try:
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        path = os.path.join(file_path, file_name + ".csv")

        logger.info("Saving file in {}".format(path))

        df.to_csv(
            path,
            index=False,
            header=True,
            quoting=csv.QUOTE_ALL,
            encoding="utf-8",
            escapechar="\\",
            sep=",",
        )
    except Exception as e:
        stacktrace = traceback.format_exc()
        logger.error("{}".format(stacktrace))

        raise e


def transform_data(df):
    try:
        df = df[["text", "Sentiment"]]

        logger.info("Original count: {}".format(len(df.index)))

        df = df.dropna()

        df["text"] = df["text"].apply(lambda x: clean_text(x))
        df["text"] = df["text"].map(lambda x: x.strip())
        df["text"] = df["text"].replace("", np.nan)
        df["text"] = df["text"].replace(" ", np.nan)

        df["Sentiment"] = df["Sentiment"].map(lambda x: x.strip())
        df["Sentiment"] = df["Sentiment"].replace("", np.nan)
        df["Sentiment"] = df["Sentiment"].replace(" ", np.nan)

        df["Sentiment"] = df["Sentiment"].map(
            {"Negative": 0, "Neutral": 1, "Positive": 2}
        )

        df = df.dropna()

        df = df.rename(columns={"Sentiment": "labels"})

        df = df[["text", "labels"]]

        logger.info("Current count: {}".format(len(df.index)))

        return df
    except Exception as e:
        stacktrace = traceback.format_exc()
        logger.error("{}".format(stacktrace))

        raise e


if __name__ == "__main__":

    args = __read_params()

    df = extract_data(args.train, 100)

    df = transform_data(df)

    train, test = train_test_split(df, test_size=0.2)

    X_train, y_train = train["text"], train["labels"].values
    X_test, y_test = test["text"], test["labels"].values

    # Create a bag-of-words vectorizer
    vectorizer = CountVectorizer()

    # Fit the vectorizer to your dataset
    vectorizer.fit(X_train)

    # Convert the input strings to numerical representations
    X_train = vectorizer.transform(X_train).toarray()
    X_test = vectorizer.transform(X_test).toarray()

    X_train, y_train = torch.from_numpy(X_train).type(torch.float32), torch.from_numpy(
        y_train
    )
    X_test, y_test = torch.from_numpy(X_test).type(torch.float32), torch.from_numpy(
        y_test
    )

    model = MulticlassClassifier(X_train.shape[1], 20, 3)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    logger.info("Start training: {}".format(strftime("%Y-%m-%d %H:%M:%S", gmtime())))

    for epoch in range(args.epochs):
        logger.info("Epoch: {}".format(epoch))
        # Forward pass: compute predicted y by passing x to the model
        y_pred = model(X_train)

        # Compute and print loss
        loss = criterion(y_pred, y_train)
        logger.info(f"Training Loss: {loss:.4f}")

        # Compute the accuracy
        _, predicted = torch.max(y_pred, dim=1)
        correct = (predicted == y_train).sum().item()
        accuracy = correct / len(y_train)
        logger.info(f"Training Accuracy: {accuracy:.4f}")

        # Zero gradients, perform a backward pass, and update the weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    logger.info("End training: {}".format(strftime("%Y-%m-%d %H:%M:%S", gmtime())))

    with torch.no_grad():
        y_pred = model(X_test)
        # Compute and print loss
        loss = criterion(y_pred, y_test)
        logger.info(f"Evaluation Loss: {loss:.4f}")

        # Compute the accuracy
        _, predicted = torch.max(y_pred, dim=1)
        correct = (predicted == y_test).sum().item()
        accuracy = correct / len(y_test)
        logger.info(f"Evaluation Accuracy: {accuracy:.4f}")

    logger.info("Save model in {}".format(args.model_dir))

    torch.save(model.state_dict(), "{}/model.pth".format(args.model_dir))

    logger.info("Save vectorizer in {}".format(args.model_dir))

    with open("{}/vectorizer.pkl".format(args.model_dir), "wb") as f:
        pickle.dump(vectorizer, f)
