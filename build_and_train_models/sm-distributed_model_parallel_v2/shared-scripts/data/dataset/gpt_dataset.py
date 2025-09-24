"""Data pipeline."""
import gzip
import json
from io import BytesIO
from typing import List, Tuple, TypeVar

import numpy as np
import torch
import torch.distributed as dist
from logging_utils import get_logger

logger = get_logger()
T_co = TypeVar("T_co", covariant=True)


def chunks(l, n):
    """Yield n number of striped chunks from l."""
    for i in range(0, n):
        yield l[i::n]


###### Load GPT pretraining data ######
class GPTPretrainingDataset(torch.utils.data.Dataset):
    """GPT Pretraining Dataset."""

    def __init__(
        self,
        input_paths: List[str],
        max_sequence_length=None,
        zipped=True,
    ):
        self.input_paths = input_paths
        self.max_sequence_length = max_sequence_length
        self.zipped = zipped
        self.drop_last = True
        self.input_data = []
        self.num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.__read_examples(self.input_paths)

    def __read_examples(self, paths: List[str]):
        for path in paths:
            self.input_data = []
            # 1 below:  each item of an S3Dataset object is a pair
            # The 0th element is a string for S3 object address
            # The 1st element is binary data
            if isinstance(path, tuple):
                filepath = path[0]
                fileobj = BytesIO(path[1])
            else:
                fileobj = path

            if self.zipped:
                with gzip.open(fileobj, "rt") as f:
                    self.input_data = [ln for _, ln in enumerate(f, 1)]
            else:
                with open(fileobj, "r") as f:
                    self.input_data = [ln for ln in f]
            if dist.get_rank() == 0:
                logger.debug(f"Read {len(self.input_data)} sequences from file")

    def __len__(self) -> int:
        return len(self.input_data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        obj = json.loads(self.input_data[index])
        iids = torch.tensor(obj["input_ids"], dtype=torch.long)
        attns = torch.tensor(obj["attention_mask"], dtype=torch.long)
        self.actual_sequence_length = len(obj["input_ids"])

        if self.actual_sequence_length > self.max_sequence_length:
            s_idx = np.random.randint(0, self.actual_sequence_length - self.max_sequence_length)
            e_idx = s_idx + self.max_sequence_length
            iids = iids[s_idx:e_idx]
            attns = attns[s_idx:e_idx]
        return iids, attns
