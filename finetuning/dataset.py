"""Dataset utilities for TabPFN finetuning."""

from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class TabularDataset(Dataset):
    """Dataset for tabular data."""

    def __init__(
        self, X: np.ndarray, y: np.ndarray, transform: Optional[callable] = None
    ) -> None:
        """Initialize dataset.

        Args:
            X: Feature matrix
            y: Target vector
            transform: Optional transform to apply to features
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.transform = transform

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get dataset item.

        Args:
            idx: Item index

        Returns:
            Tuple of (features, target)
        """
        X = self.X[idx]
        if self.transform:
            X = self.transform(X)
        return X, self.y[idx]


class TabPFNDataModule(LightningDataModule):
    """PyTorch Lightning data module for TabPFN finetuning."""

    def __init__(
        self,
        data_path: str,
        target_column: str,
        batch_size: int = 64,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
    ) -> None:
        """Initialize data module.

        Args:
            data_path: Path to CSV data file
            target_column: Name of target column
            batch_size: Batch size for dataloaders
            test_size: Fraction of data to use for test set
            val_size: Fraction of data to use for validation set
            random_state: Random seed for data splitting
        """
        super().__init__()
        self.data_path = data_path
        self.target_column = target_column
        self.batch_size = batch_size
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

        self.train_dataset: Optional[TabularDataset] = None
        self.val_dataset: Optional[TabularDataset] = None
        self.test_dataset: Optional[TabularDataset] = None

    def prepare_data(self) -> None:
        """Download or prepare data if needed."""
        # Nothing to do for local CSV files
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up datasets.

        Args:
            stage: Optional stage parameter ("fit", "test", "predict")
        """
        # Load data
        data = pd.read_csv(self.data_path)
        X = data.drop(columns=[self.target_column]).values
        y = data[self.target_column].values

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        if self.val_size > 0:
            val_size_adjusted = self.val_size / (1 - self.test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=val_size_adjusted, random_state=self.random_state
            )
            self.val_dataset = TabularDataset(X_val, y_val)

        self.train_dataset = TabularDataset(X_train, y_train)
        self.test_dataset = TabularDataset(X_test, y_test)

    def train_dataloader(self) -> DataLoader:
        """Get training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
        )

    def val_dataloader(self) -> DataLoader:
        """Get validation dataloader."""
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
        )

    def test_dataloader(self) -> DataLoader:
        """Get test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
        )