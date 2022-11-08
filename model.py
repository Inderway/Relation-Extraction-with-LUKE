# The main model for relation extraction
# created by wei
# Nov 8, 2022

from transformers import LukeForEntityPairClassification, AdamW
import pytorch_lightning as pl
import torch

class LUKE(pl.LightningModule):

    def __init__(self, num_labels):
        super().__init__()
        self.model = LukeForEntityPairClassification.from_pretrained("studio-ousia/luke-base", num_labels=num_labels)

    def forward(self, input_ids, entity_ids, entity_position_ids, attention_mask, entity_attention_mask):     
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, entity_ids=entity_ids, 
                             entity_attention_mask=entity_attention_mask, entity_position_ids=entity_position_ids)
        return outputs
    
    def common_step(self, batch, batch_idx):
        labels = batch['label']
        del batch['label']
        outputs = self(**batch)
        logits = outputs.logits

        criterion = torch.nn.CrossEntropyLoss() # multi-class classification
        loss = criterion(logits, labels)
        predictions = logits.argmax(-1)
        correct = (predictions == labels).sum().item()
        accuracy = correct/batch['input_ids'].shape[0]

        return loss, accuracy
      
    def training_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)     
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)
        self.log("training_accuracy", accuracy)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)     
        self.log("validation_loss", loss, on_epoch=True)
        self.log("validation_accuracy", accuracy, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)     

        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=5e-5)
        return optimizer

