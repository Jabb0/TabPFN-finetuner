# TabPFN Finetuning

This project provides utilities for finetuning TabPFN models on custom datasets using PyTorch Lightning.

## Installation

1. Clone this repository
2. Make sure TabPFN is available at `../TabPFN` relative to this project

```bash
poetry install
```

## Usage

```python
from finetuning.trainer import TabPFNFinetuner

finetuner = TabPFNFinetuner(
    dataset_path="path/to/dataset.csv",
    target_column="target",
    max_epochs=10
)
finetuner.train()
finetuner.save_model("finetuned_model.pt")
```

## Development
