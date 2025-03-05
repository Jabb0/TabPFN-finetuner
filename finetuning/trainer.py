"""Trainer utilities for TabPFN finetuning."""

from typing import Dict, Optional, Union

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger

from finetuning.dataset import TabPFNDataModule
from finetuning.model import TabPFNModule


class TabPFNFinetuner:
    """TabPFN model finetuner."""

    def __init__(
        self,
        dataset_path: str,
        target_column: str,
        batch_size: int = 64,
        max_epochs: int = 50,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        freeze_encoder: bool = False,
        patience: int = 10,
        gpus: Optional[Union[int, str]] = None,
    ) -> None:
        """Initialize finetuner.

        Args:
            dataset_path: Path to CSV data file
            target_column: Name of target column
            batch_size: Batch size for training
            max_epochs: Maximum number of training epochs
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            freeze_encoder: Whether to freeze encoder weights
            patience: Patience for early stopping
            gpus: GPUs to use for training (None for CPU)
        """
        self.dataset_path = dataset_path
        self.target_column = target_column
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.freeze_encoder = freeze_encoder
        self.patience = patience
        self.gpus = gpus

        # Initialize data module
        self.data_module = TabPFNDataModule(
            data_path=dataset_path,
            target_column=target_column,
            batch_size=batch_size,
        )

        # Initialize model
        self.model = TabPFNModule(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            freeze_encoder=freeze_encoder,
        )

        # Initialize trainer
        self.logger = TensorBoardLogger("logs", name="tabpfn")
        self.callbacks = [
            ModelCheckpoint(
                monitor="val_loss",
                filename="tabpfn-{epoch:02d}-{val_loss:.4f}",
                save_top_k=3,
                mode="min",
            ),
            EarlyStopping(
                monitor="val_loss",
                patience=patience,
                mode="min",
                verbose=True,
            ),
            LearningRateMonitor(logging_interval="epoch"),
        ]

        self.trainer = pl.Trainer(
            max_epochs=max_epochs,
            gpus=gpus,
            logger=self.logger,
            callbacks=self.callbacks,
            log_every_n_steps=10,
            deterministic=True,
        )

    def train(self) -> None:
        """Train the model."""
        self.trainer.fit(self.model, self.data_module)

    def test(self) -> Dict:
        """Test the model.

        Returns:
            Test results
        """
        return self.trainer.test(self.model, self.data_module)[0]

    def save_model(self, model_path: str) -> None:
        """Save the model.

        Args:
            model_path: Path to save model
        """
        self.trainer.save_checkpoint(model_path)