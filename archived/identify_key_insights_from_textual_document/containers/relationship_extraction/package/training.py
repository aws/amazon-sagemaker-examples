import argparse
import os
import json
from pathlib import Path
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch
from torch.utils.data import DataLoader

from package.data.tokenizers import RelationshipTokenizer
from package.data.label_encoders import LabelEncoder
from package.data.semeval import label_set
from package.data.dataset import RelationStatementDataset
from package.models import RelationshipEncoderLightningModule




def parse_args(sys_args):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pretrained-model",
        type=str,
        default="bert-base-uncased"
    )
    parser.add_argument(
        "--max-epoch",
        type=int,
        default=5
    )    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.0007
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16
    )    
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=3
    )       
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0
    )     
    parser.add_argument(
        "--accumulate-grad-batches",
        type=int,
        default=2
    )       
    parser.add_argument(
        "--gradient-clip-val",
        type=float,
        default=1.0
    )       
    parser.add_argument(
        "--gpus",
        type=int,
        default=os.environ.get("SM_NUM_GPUS", 0)
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ.get("SM_MODEL_DIR")
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.environ.get("SM_OUTPUT_DATA_DIR")
    )
    parser.add_argument(
        "--train-data-dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN"),
    )
    parser.add_argument(
        "--validation-data-dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_VALIDATION")
    )

    args, _ = parser.parse_known_args(sys_args)
    return args

    
def train_fn(args):
    print(args)
    
    # load tokenizer
    tokenizer = RelationshipTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.pretrained_model,
        contains_entity_tokens=False
    )
    tokenizer.save(file_path=Path(args.model_dir, 'tokenizer.json'), pretty=True)
    
    # load data
    train_file_path = Path(args.train_data_dir, 'train.txt')
    test_file_path = Path(args.validation_data_dir, 'validation.txt')
    
    # construct label encoder
    labels = list(label_set(train_file_path))
    label_encoder = LabelEncoder.from_str_list(sorted(labels))
    print('Using the following label encoder mappings:\n\n', label_encoder)
    label_encoder.save(file_path=str(Path(args.model_dir, 'label_encoder.json')))
    
    # prepare datasets
    model_size = 512
    tokenizer.set_truncation(model_size)
    tokenizer.set_padding(model_size)
    train_dataset = RelationStatementDataset(
        file_path=train_file_path,
        tokenizer=tokenizer,
        label_encoder=label_encoder
    )
    test_dataset = RelationStatementDataset(
        file_path=test_file_path,
        tokenizer=tokenizer,
        label_encoder=label_encoder
    )

    batch_size = args.batch_size
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=4
    )
    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=4
    )
    
    # create model
    relationship_encoder = RelationshipEncoderLightningModule(
        args.pretrained_model,
        tokenizer,
        label_encoder,
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='valid_loss',
        filepath=str(Path(args.model_dir, 'model')),
        verbose=True,
        mode="min",
    )
    
    # train model
    trainer = Trainer(
        default_root_dir=args.output_dir,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        max_epochs=args.max_epoch,
        weights_summary='full',
        gpus=args.gpus,
        checkpoint_callback=checkpoint_callback,
        callbacks=[EarlyStopping(monitor='valid_accuracy', patience=args.early_stop_patience, mode='max', verbose=True)],
        fast_dev_run=False,
    )
    
    trainer.fit(relationship_encoder, train_dataloader, test_dataloader)
    
    with open(str(Path(args.model_dir, 'pretrained_model_info.json')), 'w') as f:
        json.dump(
            {"pretrained_model": args.pretrained_model},
            f
        )