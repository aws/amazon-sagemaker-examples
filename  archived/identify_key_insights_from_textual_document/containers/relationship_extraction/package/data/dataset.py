import torch
from package.data.label_encoders import LabelEncoder
from package.data.semeval import parse_file
from tokenizers import Tokenizer
from torch.utils.data.dataset import Dataset


class RelationStatementDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: Tokenizer, label_encoder: LabelEncoder) -> None:
        self.relationships = parse_file(file_path)
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder

    def __getitem__(self, idx):
        relationship = self.relationships[idx]
        statement = str(relationship.statement)
        label_id = self.label_encoder.str_to_id(relationship.directed_label)
        entity_one = relationship.entity_one
        entity_two = relationship.entity_two
        encoding = self.tokenizer.encode(
            sequence=statement,
            entity_one_start=entity_one.start_char,
            entity_one_end=entity_one.end_char,
            entity_two_start=entity_two.start_char,
            entity_two_end=entity_two.end_char,
        )
        return {
            "token_ids": torch.tensor(encoding["ids"]),
            "attention_mask": torch.tensor(encoding["attention_mask"]),
            "special_tokens_mask": torch.tensor(encoding["special_tokens_mask"]),
            "label_id": torch.tensor(label_id),
        }

    def __len__(self):
        return len(self.relationships)
