"""Tests for the dataset module."""

import os
import tempfile
from typing import Tuple

import numpy as np
import pandas as pd
import pytest
import torch

from finetuning.dataset import TabPFNDataModule, TabularDataset


@pytest.fixture
def sample_data() -> Tuple[np.ndarray, np.ndarray]:
    """Create sample data for testing.

    Returns:
        Tuple of (X, y)
    """
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    return X, y


@pytest.fixture
def sample_csv() -> str:
    """Create sample CSV file for testing.

    Returns:
        Path to CSV file
    """
    # Create data
    data = pd.DataFrame(
        np.random.rand(100, 10),
        columns=[f"feature_{i}" for i in range(10)],
    )
    data["target"] = np.random.randint(0, 2, 100)
    
    # Write to temp file
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        data.to_csv(tmp.name, index=False)
        return tmp.name


def test_tabular_dataset(sample_data: Tuple[np.ndarray, np.ndarray]) -> None:
    """Test TabularDataset.

    Args:
        sample_data: Sample data for testing
    """
    X, y = sample_data
    dataset = TabularDataset(X, y)
    
    # Check lengths
    assert len(dataset) == 100
    
    # Check item
    x, y_item = dataset[0]
    assert isinstance(x, torch.Tensor)
    assert isinstance(y_item, torch.Tensor)
    assert x.shape == (10,)
    assert y_item.shape == ()


def test_tabpfn_data_module(sample_csv: str) -> None:
    """Test TabPFNDataModule.

    Args:
        sample_csv: Path to sample CSV file
    """
    try:
        data_module = TabPFNDataModule(
            data_path=sample_csv,
            target_column="target",
            batch_size=32,
        )
        
        # Setup
        data_module.prepare_data()
        data_module.setup()
        
        # Check dataloaders
        train_dataloader = data_module.train_dataloader()
        val_dataloader = data_module.val_dataloader()
        test_dataloader = data_module.test_dataloader()
        
        assert train_dataloader is not None
        assert val_dataloader is not None
        assert test_dataloader is not None
        
        # Check batch
        batch = next(iter(train_dataloader))
        x, y = batch
        
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        assert x.dim() == 2
        assert y.dim() == 1
        assert x.shape[0] == y.shape[0]
    finally:
        # Clean up
        os.unlink(sample_csv)