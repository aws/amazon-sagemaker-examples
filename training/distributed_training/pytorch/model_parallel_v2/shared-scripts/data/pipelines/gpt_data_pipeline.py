"""Data pipeline."""
import os
from typing import List, Union

from data.dataset.gpt_dataset import GPTPretrainingDataset
from data.pipelines.data_pipeline import DataPipeline
from data.utils import is_s3_source
from logging_utils import get_logger

try:
    from awsio.python.lib.io.s3.s3dataset import S3Dataset
except ModuleNotFoundError:
    S3Dataset = None

logger = get_logger()


class GPTDataPipeline(DataPipeline):
    def __init__(
        self,
        dataset_train_path,
        train_batch_size,
        dataset_val_path=None,
        val_batch_size=None,
        start_path_index=0,
        use_last_file_only_for_valid=False,
        sequence_length=2048,
        dataset_type="gpt",
        zipped_data=False,
        seed=1234,
        num_workers=0,
        resume_from_sequence_number=0,
        dp_rank=0,
        dp_size=1,
        shuffle=False,
    ):
        super().__init__(
            train_batch_size,
            val_batch_size=val_batch_size,
            seed=seed,
            num_workers=num_workers,
            resume_from_sequence_number=resume_from_sequence_number,
            dp_rank=dp_rank,
            dp_size=dp_size,
            shuffle=shuffle,
        )
        self.sequence_length = sequence_length
        self.train_paths = self.get_train_paths(
            dataset_type, dataset_train_path, zipped_data=zipped_data
        )
        self.cur_train_path = start_path_index
        self.zipped_data = zipped_data
        self.start_path_index = start_path_index
        # needs to be called explicitly
        # self._create_train_dataset()
        if val_batch_size and dataset_val_path:
            self.val_paths = self.get_val_paths(
                dataset_type, dataset_val_path, zipped_data=zipped_data
            )
            self.use_last_file_only_for_valid = use_last_file_only_for_valid
            self._create_val_dataset()

    def _create_val_dataset(self):
        self.val_dataset = GPTPretrainingDataset(
            self.val_paths if not self.use_last_file_only_for_valid else [self.val_paths[-1]],
            max_sequence_length=self.sequence_length,
            zipped=self.zipped_data,
        )
        self.val_dataloader = self._create_dataloader(self.val_dataset, self.val_batch_size)

    def increment_path_in_epoch(self):
        self.cur_train_path += 1
        if self.cur_train_path >= len(self.train_paths):
            self.cur_train_path = 0
            return False
        # returns if cycled through to next epoch
        return True

    def create_train_dataset(self):
        self.train_dataset = GPTPretrainingDataset(
            self.train_paths[self.cur_train_path : self.cur_train_path + 1],
            max_sequence_length=self.sequence_length,
            zipped=self.zipped_data,
        )
        self.train_dataloader = self._create_dataloader(self.train_dataset, self.train_batch_size)

    def get_train_paths(
        self, data_type, training_dir, zipped_data=False
    ) -> Union[List[str], "S3Dataset"]:
        if data_type == "bert":
            if is_s3_source(training_dir):
                raise ValueError("Unsupported BERT data from s3")
            train_paths = sorted(
                [
                    os.path.join(training_dir, p)
                    for p in os.listdir(training_dir)
                    if os.path.isfile(os.path.join(training_dir, p)) and "training" in p
                ]
            )
        elif data_type == "gpt":
            if zipped_data > 0:
                file_extension = ".json.gz"
            else:
                file_extension = ".json"
            if is_s3_source(training_dir):
                assert S3Dataset, "awsio package needs to be installed"
                train_paths = S3Dataset(training_dir)
            else:
                train_paths = sorted(
                    [
                        os.path.join(training_dir, p)
                        for p in os.listdir(training_dir)
                        if p.endswith(file_extension)
                    ]
                )
        else:
            raise NotImplementedError

        return train_paths

    def get_val_paths(
        self, data_type, test_dir, zipped_data=False
    ) -> Union[List[str], "S3Dataset"]:
        if data_type == "bert":
            if is_s3_source(test_dir):
                raise ValueError("Unsupported BERT data from s3")
            val_paths = sorted(
                [
                    os.path.join(test_dir, p)
                    for p in os.listdir(test_dir)
                    if os.path.isfile(os.path.join(test_dir, p)) and "testing" in p
                ]
            )
        elif data_type == "gpt":
            if zipped_data > 0:
                file_extension = ".json.gz"
            else:
                file_extension = ".json"
            if is_s3_source(test_dir):
                assert S3Dataset, "awsio package needs to be installed"
                val_paths = S3Dataset(test_dir)
            else:
                val_paths = sorted(
                    [
                        os.path.join(test_dir, p)
                        for p in os.listdir(test_dir)
                        if p.endswith(file_extension)
                    ]
                )
        else:
            raise NotImplementedError
        return val_paths

    def get_batch(self, data):
        input_ids, mask = data
        return input_ids, mask, input_ids

    def get_val_batch(self, data):
        input_ids, mask = data
        return input_ids, mask
