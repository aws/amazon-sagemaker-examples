from data.dataset.dummy_dataset import DummyDataset
from data.pipelines import DataPipeline


class DummyDataPipeline(DataPipeline):
    def __init__(
        self,
        vocabulary_size,
        train_batch_size,
        sequence_length,
        val_batch_size=None,
        data_type="gpt",
    ):
        super().__init__(
            train_batch_size=train_batch_size,
        )
        self.vocab_size = vocabulary_size
        self.seq_length = sequence_length
        self.train_dataset = DummyDataset(
            data_type=data_type, vocabulary_size=vocabulary_size, seqlen=sequence_length
        )
        self.train_dataloader = self._create_dataloader(self.train_dataset, self.train_batch_size)

        if val_batch_size:
            self.val_dataset = DummyDataset(
                data_type=data_type, vocabulary_size=vocabulary_size, seqlen=sequence_length
            )
            self.val_dataloader = self._create_dataloader(self.val_dataset, self.val_batch_size)

    def get_batch(self, data):
        return data[0], data[1], data[0]

    def get_val_batch(self, data):
        return self.get_batch(data)
