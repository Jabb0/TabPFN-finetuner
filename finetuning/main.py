"""Command-line interface for TabPFN finetuning."""

import argparse
import os
from typing import List

from finetuning.trainer import TabPFNFinetuner


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Finetune TabPFN model")
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to CSV dataset file",
    )
    
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Name of target column in dataset",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="tabpfn_finetuned.pt",
        help="Output path for finetuned model",
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training",
    )
    
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=50,
        help="Maximum number of training epochs",
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate for optimizer",
    )
    
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="Weight decay for optimizer",
    )
    
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Freeze encoder weights",
    )
    
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Patience for early stopping",
    )
    
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="GPUs to use for training (comma-separated list)",
    )
    
    return parser.parse_args()


def main() -> None:
    """Run the main program."""
    args = parse_args()
    
    # Initialize finetuner
    finetuner = TabPFNFinetuner(
        dataset_path=args.dataset,
        target_column=args.target,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        freeze_encoder=args.freeze_encoder,
        patience=args.patience,
        gpus=args.gpus,
    )
    
    # Train model
    print(f"Training model on {args.dataset} with target {args.target}")
    finetuner.train()
    
    # Test model
    print("Testing model...")
    test_results = finetuner.test()
    print(f"Test results: {test_results}")
    
    # Save model
    print(f"Saving model to {args.output}")
    finetuner.save_model(args.output)
    print("Done!")


if __name__ == "__main__":
    main()