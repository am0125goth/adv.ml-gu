import torch.nn as nn
import pytorch_lightning as pl
from MultiHeadSlowFast_model import MultiHeadSlowFast
from pretrainedSlowFast_model import MultiHeadSlowFastPretrained
import torch

class ActionRecognition(pl.LightningModule):
    """
    Unified training module for both models
    Usage:
        # For from-scratch model:
        from MultiHeadSlowFast_model import MultiHeadSlowFast
        model = MultiHeadSlowFast(num_verbs, num_objects, num_actions)
        trainer = ActionRecognition(model, model_name="from_scratch", learning_rate=1e-3)
        
        # For pre-trained model:
        from pretrainedSlowFast_model import MultiHeadSlowFastPretrained
        model = MultiHeadSlowFastPretrained(num_verbs, num_objects, num_actions)
        trainer = ActionRecognition(model, model_name="pretrained", learning_rate=1e-4)
    """
    def __init__(self, model, model_name="model", learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.model_name = model_name
        self.criterion = nn.BCEWithLogitsLoss()
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        videos = batch['video']
        verb_labels = batch['verb_labels']
        object_labels = batch['object_labels']
        action_labels = batch['action_labels']
        
        # Forward pass
        verb_logits, object_logits, action_logits = self(videos)
        
        # Calculate losses
        verb_loss = self.criterion(verb_logits, verb_labels.float())
        object_loss = self.criterion(object_logits, object_labels.float())
        action_loss = self.criterion(action_logits, action_labels.float())
        
        total_loss = verb_loss + object_loss + action_loss
        
        # Calculate accuracies
        verb_acc = self._calculate_multilabel_accuracy(verb_logits, verb_labels)
        object_acc = self._calculate_multilabel_accuracy(object_logits, object_labels)
        action_acc = self._calculate_multilabel_accuracy(action_logits, action_labels)

        
        self.log(f'{self.model_name}/train_total_loss', total_loss, prog_bar=True)
        self.log(f'{self.model_name}/train_verb_loss', verb_loss, prog_bar=False)
        self.log(f'{self.model_name}/train_object_loss', object_loss, prog_bar=False)
        self.log(f'{self.model_name}/train_action_loss', action_loss, prog_bar=False)
        self.log(f'{self.model_name}/train_verb_acc', verb_acc, prog_bar=True)
        self.log(f'{self.model_name}/train_object_acc', object_acc, prog_bar=False)
        self.log(f'{self.model_name}/train_action_acc', action_acc, prog_bar=False)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        videos = batch['video']
        verb_labels = batch['verb_labels']
        object_labels = batch['object_labels']
        action_labels = batch['action_labels']
        
        verb_logits, object_logits, action_logits = self(videos)
        
        verb_loss = self.criterion(verb_logits, verb_labels.float())
        object_loss = self.criterion(object_logits, object_labels.float())
        action_loss = self.criterion(action_logits, action_labels.float())
        
        total_loss = verb_loss + object_loss + action_loss
        
        verb_acc = self._calculate_multilabel_accuracy(verb_logits, verb_labels)
        object_acc = self._calculate_multilabel_accuracy(object_logits, object_labels)
        action_acc = self._calculate_multilabel_accuracy(action_logits, action_labels)

        self.log(f'{self.model_name}/val_total_loss', total_loss, prog_bar=True)
        self.log(f'{self.model_name}/val_verb_loss', verb_loss, prog_bar=False)
        self.log(f'{self.model_name}/val_object_loss', object_loss, prog_bar=False)
        self.log(f'{self.model_name}/val_action_loss', action_loss, prog_bar=False)
        self.log(f'{self.model_name}/val_verb_acc', verb_acc, prog_bar=True)
        self.log(f'{self.model_name}/val_object_acc', object_acc, prog_bar=False)
        self.log(f'{self.model_name}/val_action_acc', action_acc, prog_bar=False)
        
        return total_loss
        
    def test_step(self, batch, batch_idx):
        videos = batch['video']
        verb_labels = batch['verb_labels']
        object_labels = batch['object_labels']
        action_labels = batch['action_labels']
        
        verb_logits, object_logits, action_logits = self(videos)
        
        verb_loss = self.criterion(verb_logits, verb_labels.float())
        object_loss = self.criterion(object_logits, object_labels.float())
        action_loss = self.criterion(action_logits, action_labels.float())
        
        total_loss = verb_loss + object_loss + action_loss
        
        verb_acc = self._calculate_multilabel_accuracy(verb_logits, verb_labels)
        object_acc = self._calculate_multilabel_accuracy(object_logits, object_labels)
        action_acc = self._calculate_multilabel_accuracy(action_logits, action_labels)

        self.log(f'{self.model_name}/test_total_loss', total_loss)
        self.log(f'{self.model_name}/test_verb_loss', verb_loss)
        self.log(f'{self.model_name}/test_object_loss', object_loss)
        self.log(f'{self.model_name}/test_action_loss', action_loss)
        self.log(f'{self.model_name}/test_verb_acc', verb_acc)
        self.log(f'{self.model_name}/test_object_acc', object_acc)
        self.log(f'{self.model_name}/test_action_acc', action_acc)
        
        return total_loss

    def _calculate_multilabel_accuracy(self, logits, labels):
        """Calculate accuracy for multi-label classification"""
        predictions = (torch.sigmoid(logits) > 0.5).float()
        correct = (predictions == labels).float()
        return correct.mean()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=1e-4 
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': f'{self.model_name}/val_total_loss'
            }
        }