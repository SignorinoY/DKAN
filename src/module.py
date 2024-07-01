from __future__ import annotations

import lightning.pytorch as pl
import torch
import torch.nn as nn
from torchmetrics.functional import accuracy

from src.config import DataModuleConfig, ModuleConfig
from src.dkan import DKAN
from src.kan import KAN


class Classifier(pl.LightningModule):

    def __init__(
        self,
        model_name: str = ModuleConfig.model_name,
        hidden_dim: int = ModuleConfig.hidden_dim,
        input_dim: int = DataModuleConfig.input_dim,
        num_classes: int = DataModuleConfig.num_classes,
        lr: float = ModuleConfig.lr,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        if model_name == "DKAN":
            self.model = DKAN([input_dim, hidden_dim, num_classes])
        elif model_name == "KAN":
            self.model = KAN([input_dim, hidden_dim, num_classes], grid_size=7)
        else:
            self.model = nn.Sequential(
                nn.Linear(input_dim, hidden_dim * 12),
                nn.ReLU(),
                nn.Linear(hidden_dim * 12, num_classes)
            )

        self.num_classes = num_classes
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = accuracy

    def forward(self, x):
        x = x.view(x.size(0), -1).float()
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        image, label = batch
        logits = self.forward(image)
        loss = self.criterion(logits,label)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        image, label = batch
        logits = self.forward(image)
        loss = self.criterion(logits,label)
        self.log("val_loss", loss, prog_bar=True)
        predicted_labels = torch.argmax(logits, 1)
        acc = self.accuracy(
            predicted_labels, label,
            num_classes=self.num_classes,
            task="multiclass",
        )
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        image, label = batch
        logits = self.forward(image)
        loss = self.criterion(logits,label)
        self.log("test_loss", loss)
        predicted_labels = torch.argmax(logits, 1)
        acc = self.accuracy(
            predicted_labels, label,
            num_classes=self.num_classes,
            task="multiclass",
        )
        self.log("test_acc", acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val_loss",
                "frequency": 1,
                "name": "lr"
            }
        }


class Regressor(pl.LightningModule):

    def __init__(
        self,
        model_name: str = ModuleConfig.model_name,
        hidden_dim: int = ModuleConfig.hidden_dim,
        input_dim: int = DataModuleConfig.input_dim,
        lr: float = ModuleConfig.lr,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        if model_name == "DKAN":
            self.model = DKAN([input_dim, hidden_dim, 1])
        elif model_name == "KAN":
            self.model = KAN([input_dim, hidden_dim, 1], grid_size=8)
        else:
            self.model = nn.Sequential(
                nn.Linear(input_dim, hidden_dim * 12),
                nn.ReLU(),
                nn.Linear(hidden_dim * 12, 1)
            )

        self.lr = lr
        self.criterion = nn.MSELoss()
        self.accuracy = accuracy

    def forward(self, x):
        x = x.view(x.size(0), -1).float()
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self.forward(x)
        loss = self.criterion(y_pred, y_true)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self.forward(x)
        loss = self.criterion(y_pred, y_true)
        self.log("val_loss", loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self.forward(x)
        loss = self.criterion(y_pred, y_true)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val_loss",
                "frequency": 1,
                "name": "lr"
            }
        }
