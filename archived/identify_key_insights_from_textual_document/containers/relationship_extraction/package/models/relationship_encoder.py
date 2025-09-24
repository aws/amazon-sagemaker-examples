import torch
from transformers import BertModel


class RelationshipEncoder(torch.nn.Module):
    def __init__(self, vocab_length, num_classes, entity_one_start_token_id, entity_two_start_token_id):
        super(RelationshipEncoder, self).__init__()
        self.entity_one_start_token_id = entity_one_start_token_id
        self.entity_two_start_token_id = entity_two_start_token_id
        self.text_encoder = BertModel.from_pretrained(pretrained_model_name_or_path="bert-base-uncased")
        self.text_encoder.resize_token_embeddings(vocab_length)
        self.layer_norm = torch.nn.LayerNorm(self.text_encoder.config.hidden_size * 2)
        self.linear = torch.nn.Linear(self.text_encoder.config.hidden_size * 2, num_classes)

    def forward(self, token_ids, attention_mask):
        output = self.text_encoder(input_ids=token_ids, attention_mask=attention_mask, return_dict=True)
        last_hidden_state = output["last_hidden_state"]
        entity_one_mask = (token_ids == self.entity_one_start_token_id).int()
        entity_one_embedding = (entity_one_mask.unsqueeze(dim=-1) * last_hidden_state).sum(dim=1)
        entity_two_mask = (token_ids == self.entity_two_start_token_id).int()
        entity_two_embedding = (entity_two_mask.unsqueeze(dim=-1) * last_hidden_state).sum(dim=1)
        relationship_embedding = torch.cat([entity_one_embedding, entity_two_embedding], dim=1)
        relationship_embedding_norm = self.layer_norm(relationship_embedding)
        logits = self.linear(relationship_embedding_norm)
        return logits
