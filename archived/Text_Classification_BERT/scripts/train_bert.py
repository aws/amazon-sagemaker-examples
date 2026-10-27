import argparse
import logging
import os
from typing import Any

import pandas as pd
import time
import torch
import numpy as np
import pytorch_lightning as pl

from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForSequenceClassification, BertTokenizer, get_linear_schedule_with_warmup

### implementation - optional
from smart_sifting.data_model.data_model_interface import SiftingBatch, SiftingBatchTransform
from smart_sifting.data_model.list_batch import ListBatch
t


#### interface
from smart_sifting.dataloader.sift_dataloader import SiftingDataloader
from smart_sifting.loss.abstract_sift_loss_module import Loss
from smart_sifting.metrics.lightning import TrainingMetricsRecorder


"""
This is the sagemaker entrypoint script that trains a BERT model on a public CoLA dataset using PyTorch Lightning modules.

Note that this script should only be used as a Sagemaker entry point. It assumes the data is saved in an S3 location
thats specified by the "SM_CHANNEL_DATA" environment variable set by Sagemaker. It also assumes that there are the following
parameters passed in as arguments:
num_nodes - the number of nodes, or instances, that the training script should run on
dev_mode - flag determining whether or not to turn on dev mode. This is useful for debugging runs.

See run_bert_ptl.py for the Sagemaker PyTorch Estimator that sets these parameters.

The data processing and training code is adapted from https://www.kaggle.com/code/hassanamin/bert-pytorch-cola-classification/notebook
"""

RANDOM_SEED = 7

# Setting up logger for this module
logger = logging.getLogger(__name__)

class BertLoss(Loss):
    """
    This is an implementation of the Loss interface for the BERT model
    required for Smart Sifting. Use Cross-Entropy loss with 2 classes
    """
    def __init__(self):
        self.celoss = torch.nn.CrossEntropyLoss(reduction='none')

    def loss(
            self,
            model: torch.nn.Module,
            transformed_batch: SiftingBatch,
            original_batch: Any = None,
    ) -> torch.Tensor:
        # get original batch onto model device. Note that we are assuming the model is on the right device here already
        # Pytorch lightning takes care of this under the hood with the model thats passed in.
        # TODO: ensure batch and model are on the same device in SiftDataloader so that the customer
        #  doesn't have to implement this
        device = next(model.parameters()).device
        batch = [t.to(device) for t in original_batch]

        # compute loss
        outputs = model(batch)
        return self.celoss(outputs.logits, batch[2])


class BertListBatchTransform(SiftingBatchTransform):
    """
    This is an implementation of the data transforms for the BERT model
    required for Smart Sifting. Transform to and from ListBatch
    """
    def transform(self, batch: Any):
        inputs = []
        for i in range(len(batch[0])):
            inputs.append((batch[0][i], batch[1][i]))

        labels = batch[2].tolist()  # assume the last one is the list of labels
        return ListBatch(inputs, labels)

    def reverse_transform(self, list_batch: ListBatch):
        inputs = list_batch.inputs
        input_ids = [iid for (iid, _) in inputs]
        masks = [mask for (_, mask) in inputs]
        a_batch = [torch.stack(input_ids), torch.stack(masks), torch.tensor(list_batch.labels, dtype=torch.int64)]
        return a_batch


class ColaDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, model: torch.nn.Module, log_batches: bool):
        super().__init__()
        self.batch_size = batch_size
        self.model = model
        self.log_batches = log_batches

    def setup(self, stage: str) -> None:
        """
        Loads the data from s3, splits it into multi-batches.
        This logic is dataset specific.
        """

        logger.info(f"Preprocessing CoLA dataset")

        # The environment variable for the path to find the dataset is SM_CHANNEL_{channel_name}, which should match
        # with the channel name specified in the estimator in `run_bert_ptl.py`
        data_path = os.environ["SM_CHANNEL_DATA"]
        dataframe = pd.read_csv(
            f"{data_path}/train.tsv",
            sep="\t"
        )

        # Split dataframes (Note: we use scikitlearn here because pytorch random_split doesn't work as intended - theres
        # a bug when we pass in proportions to random_split (https://stackoverflow.com/questions/74327447/how-to-use-random-split-with-percentage-split-sum-of-input-lengths-does-not-equ)
        # and we get a KeyError when iterating through the resulting split datasets)
        logger.info(f"Splitting dataframes into train, val, and test")
        train_df, test_df = train_test_split(dataframe, train_size=0.9, random_state=RANDOM_SEED)
        train_df, val_df = train_test_split(train_df, train_size=0.9, random_state=RANDOM_SEED)

        # Finally, transform the dataframes into PyTorch datasets
        logger.info(f"Transforming dataframes into datasets")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        max_sentence_length = 128

        self.train = self._transform_to_dataset(train_df, tokenizer, max_sentence_length)
        self.val = self._transform_to_dataset(val_df, tokenizer, max_sentence_length)
        self.test = self._transform_to_dataset(test_df, tokenizer, max_sentence_length)

        logger.info("Done preprocessing CoLA dataset")

    def train_dataloader(self):
        sift_config = RelativeProbabilisticSiftConfig(
            beta_value=3,
            loss_history_length=500,
            loss_based_sift_config=LossConfig(
                 sift_config=SiftingBaseConfig(sift_delay=10)
            )
        )
        
        return SiftingDataloader(
            sift_config = sift_config,
            orig_dataloader=DataLoader(self.train, self.batch_size, shuffle=True),
            loss_impl=BertLoss(),
            model=self.model,
            batch_transforms=BertListBatchTransform()
        )       

    def val_dataloader(self):
        return DataLoader(self.val, self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.test, self.batch_size)

    def _transform_to_dataset(self, dataframe: pd.DataFrame, tokenizer, max_sentence_length):
        sentences = dataframe.sentence.values
        labels = dataframe.label.values

        input_ids = []
        for sent in sentences:
            encoded_sent = tokenizer.encode(sent, add_special_tokens=True)
            input_ids.append(encoded_sent)

        # pad shorter sentences
        input_ids_padded = []
        for i in input_ids:
            while len(i) < max_sentence_length:
                i.append(0)
            input_ids_padded.append(i)
        input_ids = input_ids_padded

        # mask; 0: added, 1: otherwise
        attention_masks = []
        # For each sentence...
        for sent in input_ids:
            att_mask = [int(token_id > 0) for token_id in sent]
            attention_masks.append(att_mask)

        # convert to PyTorch data types.
        inputs = torch.tensor(input_ids)
        labels = torch.tensor(labels)
        masks = torch.tensor(attention_masks)

        return TensorDataset(inputs, masks, labels)


class BertLitModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.model = self._create_model()
        self.celoss = torch.nn.CrossEntropyLoss()

    def forward(self, batch):
        return self.model(batch[0], token_type_ids=None, attention_mask=batch[1])

    def training_step(self, batch, batch_idx):
        # Forward Pass
        outputs = self(batch)
        loss = self.celoss(outputs.logits, batch[2])
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        outputs = self(batch)

        # Move tensors to CPU
        logits = outputs.logits.detach().cpu().numpy()
        label_ids = batch[2].to('cpu').numpy()

        # compute accuracy
        acc = self._flat_accuracy(logits, label_ids)

        if stage:
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        logger.info("Initializing AdamW optimizer")
        optimizer = AdamW(
            self.model.parameters(),
            lr=2e-5,  # args.learning_rate - default is 5e-5, this script has 2e-5
            eps=1e-8,  # args.adam_epsilon - default is 1e-8.
        )

        # Create the learning rate scheduler.
        logger.info("Initializing learning rate scheduler")
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0, # Default value in run_glue.py
                                                    num_training_steps=self.trainer.estimated_stepping_batches)

        return [optimizer], [scheduler]

    def _create_model(self):
        logger.info("Creating BertForSequenceClassification")
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
            num_labels=2,  # The number of output labels--2 for binary classification.
            output_attentions=False,  # Whether the model returns attentions weights.
            output_hidden_states=False,  # Whether the model returns all hidden-states.
        )

        return model

    # Function to calculate the accuracy of our predictions vs labels
    def _flat_accuracy(self, preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)


def main(args):
    # Setting up logger config
    logging.basicConfig(
        format='%(asctime)s.%(msecs)03d PID:%(process)d %(levelname)s %(module)s - %(funcName)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=args.log_level,
        force=True
    )

    pl.seed_everything(RANDOM_SEED)

    model = BertLitModule()
    # For fine-tuning BERT on a specific task, the authors recommend a batch size of 16 or 32.
    data = ColaDataModule(batch_size=32, model=model, log_batches=args.log_batches)

    trainer = pl.Trainer(
        # Authors recommend 2 - 4
        max_epochs=args.epochs,
        accelerator="auto",
        strategy="ddp",
        num_nodes=args.num_nodes,
        # Clip the norm of the gradients to 1.0. This is to help prevent the "exploding gradients" problem.
        gradient_clip_val=1,
        callbacks=[TrainingMetricsRecorder()],
    )

    trainer.fit(model, data)


# Converts string argument to boolean
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    start_time = time.perf_counter()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num_nodes", type=int, default=1, metavar="N", help="number of training nodes (default: 1)"
    )
    parser.add_argument(
        "--epochs", type=int, default=2, metavar="N", help="number of training epochs (default: 2)"
    )
    parser.add_argument(
        "--log_level", type=int, default=1, metavar="N", help="log level (default: 1)"
    )
    parser.add_argument(
        "--log_batches", type=str2bool, nargs="?", const=True, default=False, metavar="N", help="whether or not to log batches (default: False)"
    )
    args = parser.parse_args()

    main(args)
    logger.info(f"Total time : {time.perf_counter() - start_time}")