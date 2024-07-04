"""Data pipeline."""
import logging

from data.pipelines import DataPipeline
from datasets import load_from_disk
from transformers import default_data_collator

try:
    from awsio.python.lib.io.s3.s3dataset import S3Dataset
except ModuleNotFoundError:
    S3Dataset = None

logger = logging.getLogger(__file__)


class HFDataPipeline(DataPipeline):
    def __init__(
        self,
        dataset_train_path,
        train_batch_size,
        dataset_val_path=None,
        val_batch_size=None,
        seed=1234,
        num_workers=0,
        resume_from_sequence_number=0,
        val_resume_from_sequence_number=0,
        dp_rank=0,
        dp_size=1,
        shuffle=False,
    ):
        super().__init__(
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            seed=seed,
            num_workers=num_workers,
            resume_from_sequence_number=resume_from_sequence_number,
            val_resume_from_sequence_number=val_resume_from_sequence_number,
            dp_rank=dp_rank,
            dp_size=dp_size,
            shuffle=shuffle,
            collate_fn=default_data_collator,
        )
        self.train_dataset = load_from_disk(dataset_train_path)
        self.train_dataloader = self._create_dataloader(self.train_dataset, self.train_batch_size, self.resume_from_sequence_number)
        if val_batch_size and dataset_val_path:
            self.val_dataset = load_from_disk(dataset_val_path)
            self.val_dataloader = self._create_dataloader(self.val_dataset, self.val_batch_size, self.val_resume_from_sequence_number)

    def get_batch(self, data):
        return data["input_ids"], data["attention_mask"], data["labels"]

    def get_val_batch(self, data):
        return data["input_ids"], data["attention_mask"]
