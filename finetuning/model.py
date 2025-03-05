"""PyTorch Lightning model for TabPFN finetuning."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from tabpfn import TabPFNClassifier
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchmetrics


class TabPFNModule(pl.LightningModule):
    """PyTorch Lightning module for finetuning TabPFN models."""

    def __init__(
        self,
        n_estimators: int = 16,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        freeze_encoder: bool = False,
    ) -> None:
        """Initialize TabPFN module.

        Args:
            n_estimators: Number of ensemble configurations
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            freeze_encoder: Whether to freeze encoder weights
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize TabPFN model
        self.model = TabPFNClassifier(n_estimators=n_estimators)
        
        # Freeze encoder if requested
        if freeze_encoder:
            for param in self.transformer.encoder.parameters():
                param.requires_grad = False

        # TODO(fj): Make configurable and clear that this is binary classification finetuning.
        # Metrics
        self.train_acc = torchmetrics.Accuracy(
            task="binary"
        )
        self.val_acc = torchmetrics.Accuracy(
            task="binary"
        )
        self.test_acc = torchmetrics.Accuracy(
            task="binary"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features

        Returns:
            Model predictions
        """
        return self.model.predict_proba_torch(x)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict[str, Any]:
        """Training step.

        Args:
            batch: Batch of data (features, targets)
            batch_idx: Batch index

        Returns:
            Dictionary with loss and metrics
        """
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        # Log metrics
        acc = self.train_acc(logits.softmax(dim=-1), y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        
        return {"loss": loss}

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict[str, Any]:
        """Validation step.

        Args:
            batch: Batch of data (features, targets)
            batch_idx: Batch index

        Returns:
            Dictionary with loss and metrics
        """
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        # Log metrics
        acc = self.val_acc(logits.softmax(dim=-1), y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        
        return {"val_loss": loss}

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict[str, Any]:
        """Test step.

        Args:
            batch: Batch of data (features, targets)
            batch_idx: Batch index

        Returns:
            Dictionary with loss and metrics
        """
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        # Log metrics
        acc = self.test_acc(logits.softmax(dim=-1), y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        
        return {"test_loss": loss}

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers.

        Returns:
            Dictionary with optimizer and scheduler
        """
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        
        scheduler = {
            "scheduler": ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5, verbose=True
            ),
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1,
        }
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler}