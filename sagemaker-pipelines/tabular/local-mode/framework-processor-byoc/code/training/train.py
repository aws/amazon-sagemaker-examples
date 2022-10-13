from argparse import ArgumentParser
import csv
import glob
import logging
import os
import pandas as pd
from sklearn import preprocessing
import torch
from torch.utils.data import DataLoader, TensorDataset
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"

"""
    Read input data
"""
def __read_data(files_path):
    try:
        logger.info("Reading dataset from source...")

        all_files = glob.glob(os.path.join(files_path, "*.csv"))

        datasets = []

        for filename in all_files:
            data = pd.read_csv(
                filename,
                sep=',',
                quotechar='"',
                quoting=csv.QUOTE_ALL,
                escapechar='\\',
                encoding='utf-8',
                error_bad_lines=False
            )

            datasets.append(data)

        data = pd.concat(datasets, axis=0, ignore_index=True)

        data = data.dropna()

        return data
    except Exception as e:
        stacktrace = traceback.format_exc()
        logger.error("{}".format(stacktrace))

        raise e

def prepare_data(train, test):
    try:
        X_train, y_train = train.iloc[:, train.columns != 'labels'], train.iloc[:, train.columns == 'labels']
        X_test, y_test = test.iloc[:, test.columns != 'labels'], test.iloc[:, train.columns == 'labels']

        y_test = y_test.astype("int64")

        scaler = preprocessing.MinMaxScaler()

        X_train = scaler.fit_transform(X_train.values)
        X_test = scaler.fit_transform(X_test.values)

        X_train_tensor = torch.from_numpy(X_train)
        y_train_tensor = torch.from_numpy(y_train.values.ravel()).float()
        y_train_tensor = y_train_tensor.unsqueeze(1)

        X_test_tensor = torch.from_numpy(X_test)
        y_test_tensor = torch.from_numpy(y_test.values.ravel()).float()
        y_test_tensor = y_test_tensor.unsqueeze(1)

        train_ds = TensorDataset(X_train_tensor, y_train_tensor)
        test_ds = TensorDataset(X_test_tensor, y_test_tensor)

        train_dl = DataLoader(train_ds, batch_size=args.batch_size)
        test_dl = DataLoader(test_ds, batch_size=32)

        return train_dl, test_dl
    except Exception as e:
        stacktrace = traceback.format_exc()
        logger.error("{}".format(stacktrace))

        raise e

class BinaryClassifierModel(torch.nn.Module):
    def __init__(self, shape):
        super(BinaryClassifierModel, self).__init__()

        self.d1 = torch.nn.Linear(shape, 32)
        self.d2 = torch.nn.Linear(32, 64)
        self.drop = torch.nn.Dropout(0.2)
        self.output = torch.nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.d1(x))
        x = torch.relu(self.d2(x))
        x = self.drop(x)
        x = torch.sigmoid(self.output(x))

        return x

if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=1.45e-4)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))

    args = parser.parse_args()

    train = __read_data(args.train)
    test = __read_data(args.test)

    train_dl, test_dl = prepare_data(train, test)

    model = BinaryClassifierModel(train.shape[1] - 1)

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    loss_obj = torch.nn.BCELoss()

    model.train()
    train_loss = []

    for epoch in range(args.epochs):
        logger.info("Epoch {}".format(epoch + 1))

        # Within each epoch run the subsets of data = batch sizes.
        for xb, yb in train_dl:
            xb = xb.to(device)
            yb = yb.to(device)

            y_pred = model(xb.float()) # Forward Propagation

            loss = loss_obj(y_pred, yb)  # Loss Computation

            optimizer.zero_grad()  # Clearing all previous gradients, setting to zero
            loss.backward()  # Back Propagation
            optimizer.step()  # Updating the parameters

        logger.info("Training Loss: {}".format(loss.item()))
        train_loss.append(loss.item())

    torch.save(model.cpu(), os.path.join(args.model_dir, "model.pth"))