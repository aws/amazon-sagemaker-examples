import argparse
import json
import os
import random
from datetime import datetime

import dgl
import numpy as np
import torch
from dgl import model_zoo
from dgl.data.chem import Tox21
from dgl.data.utils import split_dataset
from sklearn.metrics import roc_auc_score
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader


def setup(args, seed=0):
    args["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    return args


def collate_molgraphs(data):
    """Batching a list of datapoints for dataloader."""
    smiles, graphs, labels, masks = map(list, zip(*data))

    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)
    masks = torch.stack(masks, dim=0)
    return smiles, bg, labels, masks


class EarlyStopper(object):
    def __init__(self, patience, filename=None):
        if filename is None:
            # Name checkpoint based on time
            dt = datetime.now()
            filename = "early_stop_{}_{:02d}-{:02d}-{:02d}.pth".format(
                dt.date(), dt.hour, dt.minute, dt.second
            )
            filename = os.path.join("/opt/ml/model", filename)

        self.patience = patience
        self.counter = 0
        self.filename = filename
        self.best_score = None
        self.early_stop = False

    def save_checkpoint(self, model):
        """Saves model when the metric on the validation set gets improved."""
        torch.save({"model_state_dict": model.state_dict()}, self.filename)

    def load_checkpoint(self, model):
        """Load model saved with early stopping."""
        model.load_state_dict(torch.load(self.filename)["model_state_dict"])

    def step(self, score, model):
        if (self.best_score is None) or (score > self.best_score):
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            print("EarlyStopping counter: {:d} out of {:d}".format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


class Meter(object):
    """Track and summarize model performance on a dataset for
    (multi-label) binary classification."""

    def __init__(self):
        self.mask = []
        self.y_pred = []
        self.y_true = []

    def update(self, y_pred, y_true, mask):
        """Update for the result of an iteration

        Parameters
        ----------
        y_pred : float32 tensor
            Predicted molecule labels with shape (B, T),
            B for batch size and T for the number of tasks
        y_true : float32 tensor
            Ground truth molecule labels with shape (B, T)
        mask : float32 tensor
            Mask for indicating the existence of ground
            truth labels with shape (B, T)
        """
        self.y_pred.append(y_pred.detach().cpu())
        self.y_true.append(y_true.detach().cpu())
        self.mask.append(mask.detach().cpu())

    def roc_auc_score(self):
        """Compute roc-auc score for each task.

        Returns
        -------
        list of float
            roc-auc score for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        # This assumes binary case only
        y_pred = torch.sigmoid(y_pred)
        n_tasks = y_true.shape[1]
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            scores.append(roc_auc_score(task_y_true, task_y_pred))
        return scores


def run_a_train_epoch(args, epoch, model, data_loader, loss_criterion, optimizer):
    model.train()
    train_meter = Meter()
    for batch_id, batch_data in enumerate(data_loader):
        smiles, bg, labels, masks = batch_data
        atom_feats = bg.ndata.pop(args["atom_data_field"])
        atom_feats, labels, masks = (
            atom_feats.to(args["device"]),
            labels.to(args["device"]),
            masks.to(args["device"]),
        )
        logits = model(bg, atom_feats)
        # Mask non-existing labels
        loss = (loss_criterion(logits, labels) * (masks != 0).float()).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(
            "epoch {:d}/{:d}, batch {:d}/{:d}, loss {:.4f}".format(
                epoch + 1, args["n_epochs"], batch_id + 1, len(data_loader), loss.item()
            )
        )
        train_meter.update(logits, labels, masks)
    train_score = np.mean(train_meter.roc_auc_score())
    print(
        "epoch {:d}/{:d}, training roc-auc {:.4f}".format(epoch + 1, args["n_epochs"], train_score)
    )


def run_an_eval_epoch(args, model, data_loader):
    model.eval()
    eval_meter = Meter()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, masks = batch_data
            atom_feats = bg.ndata.pop(args["atom_data_field"])
            atom_feats, labels = atom_feats.to(args["device"]), labels.to(args["device"])
            logits = model(bg, atom_feats)
            eval_meter.update(logits, labels, masks)
    return np.mean(eval_meter.roc_auc_score())


def load_sagemaker_config(args):
    file_path = "/opt/ml/input/config/hyperparameters.json"
    if os.path.isfile(file_path):
        with open(file_path, "r") as f:
            new_args = json.load(f)
            for k, v in new_args.items():
                if k not in args:
                    continue
                if isinstance(args[k], int):
                    v = int(v)
                if isinstance(args[k], float):
                    v = float(v)
                args[k] = v
    return args


def main(args):
    args = setup(args)

    dataset = Tox21()
    train_set, val_set, test_set = split_dataset(dataset, shuffle=True)
    train_loader = DataLoader(
        train_set, batch_size=args["batch_size"], shuffle=True, collate_fn=collate_molgraphs
    )
    val_loader = DataLoader(
        val_set, batch_size=args["batch_size"], shuffle=True, collate_fn=collate_molgraphs
    )
    test_loader = DataLoader(
        test_set, batch_size=args["batch_size"], shuffle=True, collate_fn=collate_molgraphs
    )

    model = model_zoo.chem.GCNClassifier(
        in_feats=args["n_input"],
        gcn_hidden_feats=[args["n_hidden"] for _ in range(args["n_layers"])],
        n_tasks=dataset.n_tasks,
        classifier_hidden_feats=args["n_hidden"],
    ).to(args["device"])
    loss_criterion = BCEWithLogitsLoss(
        pos_weight=torch.tensor(dataset.task_pos_weights).to(args["device"]), reduction="none"
    )
    optimizer = Adam(model.parameters(), lr=args["lr"])
    stopper = EarlyStopper(args["patience"])

    for epoch in range(args["n_epochs"]):
        # Train
        run_a_train_epoch(args, epoch, model, train_loader, loss_criterion, optimizer)

        # Validation and early stop
        val_score = run_an_eval_epoch(args, model, val_loader)
        early_stop = stopper.step(val_score, model)
        print(
            "epoch {:d}/{:d}, validation roc-auc {:.4f}, best validation roc-auc {:.4f}".format(
                epoch + 1, args["n_epochs"], val_score, stopper.best_score
            )
        )
        if early_stop:
            break

    stopper.load_checkpoint(model)
    test_score = run_an_eval_epoch(args, model, test_loader)
    print("Best validation score {:.4f}".format(stopper.best_score))
    print("Test score {:.4f}".format(test_score))


def parse_args():
    parser = argparse.ArgumentParser(description="GCN for Tox21")
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Number of graphs (molecules) per batch"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--n-epochs", type=int, default=100, help="Maximum number of training epochs"
    )
    parser.add_argument(
        "--atom-data-field", type=str, default="h", help="Name for storing atom features"
    )
    parser.add_argument("--n-input", type=int, default=74, help="Size for input atom features")
    parser.add_argument("--n-hidden", type=int, default=64, help="Size for hidden representations")
    parser.add_argument("--n-layers", type=int, default=2, help="Number of hidden layers")
    parser.add_argument(
        "--patience", type=int, default=10, help="Number of epochs to wait before early stop"
    )
    return parser.parse_args().__dict__


if __name__ == "__main__":
    args = parse_args()
    args = load_sagemaker_config(args)
    main(args)
