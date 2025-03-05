"""Basic example of TabPFN finetuning."""

import os
import tempfile

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from finetuning.trainer import TabPFNFinetuner


def main() -> None:
    """Run the example."""
    # Load dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target
    
    # Preprocess data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Create temporary CSV file
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create DataFrame
        train_df = pd.DataFrame(
            X_train,
            columns=[f"feature_{i}" for i in range(X_train.shape[1])],
        )
        train_df["target"] = y_train
        
        test_df = pd.DataFrame(
            X_test,
            columns=[f"feature_{i}" for i in range(X_test.shape[1])],
        )
        test_df["target"] = y_test
        
        # Save to CSV
        train_path = os.path.join(tmp_dir, "train.csv")
        test_path = os.path.join(tmp_dir, "test.csv")
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        # Initialize finetuner
        finetuner = TabPFNFinetuner(
            dataset_path=train_path,
            target_column="target",
            batch_size=32,
            max_epochs=10,
            learning_rate=1e-4,
            weight_decay=1e-5,
            freeze_encoder=False,
            patience=5,
        )
        
        # Train model
        print("Training model...")
        finetuner.train()
        
        # Test model
        print("Testing model...")
        test_results = finetuner.test()
        print(f"Test results: {test_results}")
        
        # Save model
        model_path = os.path.join(tmp_dir, "tabpfn_finetuned.pt")
        print(f"Saving model to {model_path}")
        finetuner.save_model(model_path)
        print("Done!")


if __name__ == "__main__":
    main()