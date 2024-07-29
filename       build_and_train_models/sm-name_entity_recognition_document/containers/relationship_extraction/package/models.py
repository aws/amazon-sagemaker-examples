import torch
from transformers import BertModel
import pytorch_lightning as pl


class RelationshipEncoderModule(torch.nn.Module):
    def __init__(
        self,
        pretrained_model_name,
        vocab_length,
        num_classes,
        entity_one_start_token_id,
        entity_two_start_token_id
    ):
        super(RelationshipEncoderModule, self).__init__()
        self.entity_one_start_token_id = entity_one_start_token_id
        self.entity_two_start_token_id = entity_two_start_token_id
        self.text_encoder = BertModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name
        )
        self.text_encoder.resize_token_embeddings(vocab_length)
        self.layer_norm = torch.nn.LayerNorm(self.text_encoder.config.hidden_size * 2)
        self.linear = torch.nn.Linear(self.text_encoder.config.hidden_size * 2, num_classes)

    def forward(
        self,
        token_ids,
        attention_mask
    ):
        output = self.text_encoder(
            input_ids=token_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        last_hidden_state = output['last_hidden_state']
        entity_one_mask = (token_ids == self.entity_one_start_token_id).int()
        entity_one_embedding = (entity_one_mask.unsqueeze(dim=-1) * last_hidden_state).sum(dim=1)
        entity_two_mask = (token_ids == self.entity_two_start_token_id).int()
        entity_two_embedding = (entity_two_mask.unsqueeze(dim=-1) * last_hidden_state).sum(dim=1)
        relationship_embedding = torch.cat([entity_one_embedding, entity_two_embedding], dim=1)
        relationship_embedding_norm = self.layer_norm(relationship_embedding)
        logits = self.linear(relationship_embedding_norm)
        return logits
    
    
class RelationshipEncoderLightningModule(pl.LightningModule):
    def __init__(self, pretrained_model_name, tokenizer, label_encoder, learning_rate=0.0007, weight_decay=0):
        super().__init__()
        self.model = RelationshipEncoderModule(
            pretrained_model_name=pretrained_model_name,
            vocab_length=len(tokenizer),
            num_classes=len(label_encoder),
            entity_one_start_token_id=tokenizer.entity_one_start_token_id,
            entity_two_start_token_id=tokenizer.entity_two_start_token_id
        )
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()

    def forward(self, token_ids, attention_mask):
        output = self.model(token_ids, attention_mask)
        return output

    def training_step(self, batch, batch_idx):
        token_ids = batch['token_ids']
        attention_mask = batch['attention_mask']
        label_id = batch['label_id']
        output = self.model(token_ids, attention_mask)
        pred_label = torch.argmax(output, dim=-1)
        loss = torch.nn.functional.cross_entropy(output, label_id)
        
        self.train_acc(pred_label, label_id)
        
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_accuracy', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)    
        return loss

    def validation_step(self, batch, batch_idx):
        token_ids = batch['token_ids']
        attention_mask = batch['attention_mask']
        label_id = batch['label_id']
        output = self.model(token_ids, attention_mask)
        pred_label = torch.argmax(output, dim=-1)
        loss = torch.nn.functional.cross_entropy(output, label_id)
        
        self.valid_acc(pred_label, label_id)
        
        self.log('valid_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('valid_accuracy', self.valid_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)