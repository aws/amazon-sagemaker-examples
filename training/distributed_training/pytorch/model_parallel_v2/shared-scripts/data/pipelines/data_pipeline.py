from abc import abstractmethod

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader


# Adapted from accelerate's SkipDataLoader to skip certain number of sequences instead of batches
# https://github.com/huggingface/accelerate/blob/80da9cfb09bb3cc9f1b385cb55d6b90d025a5fd9/src/accelerate/data_loader.py#L858C1-L878C28
class SkipDataLoader(DataLoader):
    """
    Subclass of a PyTorch `DataLoader` that will skip the first batches.

    Args:
        dataset (`torch.utils.data.dataset.Dataset`):
            The dataset to use to build this datalaoder.
        skip_batches (`int`, *optional*, defaults to 0):
            The number of batches to skip at the beginning.
        kwargs:
            All other keyword arguments to pass to the regular `DataLoader` initialization.
    """

    def __init__(self, *args, resume_from_sequence_number=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.resume_from_sequence_number = resume_from_sequence_number
        self.cur_seq_index = 0

    def __iter__(self):
        for batch in super().__iter__():
            num_seq = int(self.batch_size)
            if self.cur_seq_index + num_seq > self.resume_from_sequence_number % (len(self) * self.batch_size):
                yield batch
            else:
                if dist.get_rank() == 0:
                    print(
                        f"Dataloader skipping {num_seq} sequences in this batch as starting from {self.resume_from_sequence_number} sequences"
                    )
            self.cur_seq_index += num_seq


class DataPipeline:
    def __init__(
        self,
        train_batch_size,
        val_batch_size=None,
        seed=1234,
        num_workers=0,
        resume_from_sequence_number=0,
        val_resume_from_sequence_number=0,
        dp_rank=0,
        dp_size=1,
        shuffle=False,
        collate_fn=None,
    ):
        self.seed = seed
        self.num_workers = num_workers
        self.resume_from_sequence_number = resume_from_sequence_number
        self.val_resume_from_sequence_number = val_resume_from_sequence_number
        self.dp_rank = dp_rank
        self.dp_size = dp_size
        self.shuffle = shuffle
        self.collate_fn = collate_fn

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size

        self.train_dataset = None
        self.val_dataset = None
        self.train_dataloader = None
        self.val_dataloader = None

    def _create_dataloader(self, dataset, batch_size, resume_from_sequence_number):
        # TODO: set sampler.epoch to correctly shuffle across epochs, else same order will be used for
        # all epochs not relevant now as we have no epochs
        sampler = torch.utils.data.DistributedSampler(
            dataset,
            shuffle=self.shuffle,
            seed=self.seed,
            rank=self.dp_rank,
            num_replicas=self.dp_size,
            drop_last=True,
        )

        kwargs = {
            "sampler": sampler,
            "batch_size": batch_size,
            "num_workers": self.num_workers,
            "collate_fn": self.collate_fn,
            "pin_memory": True,
            "drop_last": True,
        }

        if resume_from_sequence_number > 0:
            dataloader = SkipDataLoader(
                dataset, resume_from_sequence_number=resume_from_sequence_number, **kwargs
            )
        else:
            dataloader = torch.utils.data.DataLoader(dataset, **kwargs)
        return dataloader

    @abstractmethod
    def get_batch(self, data):
        pass

    @abstractmethod
    def get_val_batch(self, data):
        pass
