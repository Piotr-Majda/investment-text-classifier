import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from tqdm import tqdm
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup

from app.core.config import settings
from app.models.pytorch.classifier import (
    InvestmentClassifierOutput,
    InvestmentTextClassifier,
)


class InvestmentTextDataset(Dataset):
    """Dataset for investment text classification."""

    def __init__(
        self,
        texts: List[str],
        risk_levels: Optional[List[str]] = None,
        horizons: Optional[List[str]] = None,
        actions: Optional[List[str]] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        max_length: int = 512,
    ):
        self.texts = texts
        self.risk_levels = risk_levels
        self.horizons = horizons
        self.actions = actions
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Create mappings from labels to indices
        self.risk_level_map = {label: i for i, label in enumerate(settings.RISK_LEVELS)}
        self.horizon_map = {
            label: i for i, label in enumerate(settings.INVESTMENT_HORIZONS)
        }
        self.action_map = {label: i for i, label in enumerate(settings.ACTIONS)}

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]

        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Convert to tensors and remove batch dimension
        item = {k: v.squeeze(0) for k, v in encoding.items()}

        # Remove token_type_ids if present (DistilBERT doesn't use them)
        if "token_type_ids" in item:
            del item["token_type_ids"]

        # Add labels if available
        if self.risk_levels is not None:
            risk_level_idx = self.risk_level_map[self.risk_levels[idx]]
            item["risk_level_labels"] = torch.tensor(risk_level_idx)

        if self.horizons is not None:
            horizon_idx = self.horizon_map[self.horizons[idx]]
            item["horizon_labels"] = torch.tensor(horizon_idx)

        if self.actions is not None:
            action_idx = self.action_map[self.actions[idx]]
            item["action_labels"] = torch.tensor(action_idx)

        return item


def load_data(data_path: str) -> pd.DataFrame:
    """Load the training data from CSV file."""
    # This function should be adapted based on your data format
    # Expected columns: text, risk_level, investment_horizon, action
    return pd.read_csv(data_path)


def evaluate_model(
    model: InvestmentTextClassifier, dataloader: DataLoader, device: torch.device
) -> Tuple[float, Dict]:
    """Evaluate the model on the provided dataloader."""
    model.eval()

    total_loss = 0
    risk_level_preds, risk_level_labels = [], []
    horizon_preds, horizon_labels = [], []
    action_preds, action_labels = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                risk_level_labels=batch.get("risk_level_labels"),
                horizon_labels=batch.get("horizon_labels"),
                action_labels=batch.get("action_labels"),
            )

            # Accumulate loss
            if outputs.loss is not None:
                total_loss += outputs.loss.item()

            # Get predictions
            risk_level_pred = (
                torch.argmax(outputs.risk_level_logits, dim=1).cpu().numpy()
            )
            horizon_pred = torch.argmax(outputs.horizon_logits, dim=1).cpu().numpy()
            action_pred = torch.argmax(outputs.action_logits, dim=1).cpu().numpy()

            # Collect predictions and labels
            risk_level_preds.extend(risk_level_pred)
            risk_level_labels.extend(batch.get("risk_level_labels").cpu().numpy())

            horizon_preds.extend(horizon_pred)
            horizon_labels.extend(batch.get("horizon_labels").cpu().numpy())

            action_preds.extend(action_pred)
            action_labels.extend(batch.get("action_labels").cpu().numpy())

    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    metrics = {
        "risk_level_accuracy": accuracy_score(risk_level_labels, risk_level_preds),
        "risk_level_f1": f1_score(
            risk_level_labels, risk_level_preds, average="weighted"
        ),
        "horizon_accuracy": accuracy_score(horizon_labels, horizon_preds),
        "horizon_f1": f1_score(horizon_labels, horizon_preds, average="weighted"),
        "action_accuracy": accuracy_score(action_labels, action_preds),
        "action_f1": f1_score(action_labels, action_preds, average="weighted"),
    }

    return avg_loss, metrics


def train_model(
    model: InvestmentTextClassifier,
    train_dataloader: DataLoader,
    eval_dataloader: Optional[DataLoader] = None,
    epochs: int = 5,
    learning_rate: float = 2e-5,
    warmup_steps: int = 0,
    save_path: str = "models/classifier.pt",
    device: Optional[torch.device] = None,
) -> InvestmentTextClassifier:
    """Train the investment text classifier model."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    # Prepare optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Training loop
    best_eval_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        # Training
        loop = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in loop:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                risk_level_labels=batch.get("risk_level_labels"),
                horizon_labels=batch.get("horizon_labels"),
                action_labels=batch.get("action_labels"),
            )

            # Backward pass
            loss = outputs.loss
            loss.backward()

            # Update parameters
            optimizer.step()
            scheduler.step()

            # Accumulate loss
            total_train_loss += loss.item()
            loop.set_postfix({"loss": loss.item()})

        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"Average train loss: {avg_train_loss:.4f}")

        # Evaluation
        if eval_dataloader is not None:
            avg_eval_loss, metrics = evaluate_model(model, eval_dataloader, device)
            print(f"Evaluation loss: {avg_eval_loss:.4f}")
            for metric_name, metric_value in metrics.items():
                print(f"{metric_name}: {metric_value:.4f}")

            # Save best model
            if avg_eval_loss < best_eval_loss:
                best_eval_loss = avg_eval_loss
                # Make sure the directory exists
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model, save_path)
                print(f"Model saved to {save_path}")

    return model


def main(args: argparse.Namespace) -> None:
    """Main function to run the training process."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer
    tokenizer_name = "distilbert-base-uncased"  # You can change this
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Make sure directories exist
    os.makedirs("models", exist_ok=True)

    # Save tokenizer for later use
    tokenizer.save_pretrained(settings.MODEL_TOKENIZER)
    print(f"Tokenizer saved to {settings.MODEL_TOKENIZER}")

    # Check if we have training data
    if os.path.exists(args.data_path):
        # Load and prepare data
        print(f"Loading data from {args.data_path}")
        df = load_data(args.data_path)

        # Create datasets
        train_dataset = InvestmentTextDataset(
            texts=df["text"].tolist(),
            risk_levels=df["risk_level"].tolist(),
            horizons=df["investment_horizon"].tolist(),
            actions=df["action"].tolist(),
            tokenizer=tokenizer,
        )

        # Create data loaders
        train_dataloader = DataLoader(
            train_dataset,
            sampler=RandomSampler(train_dataset),
            batch_size=args.batch_size,
        )

        # Create validation dataset if path provided
        eval_dataloader = None
        if args.validation_path and os.path.exists(args.validation_path):
            print(f"Loading validation data from {args.validation_path}")
            val_df = load_data(args.validation_path)
            val_dataset = InvestmentTextDataset(
                texts=val_df["text"].tolist(),
                risk_levels=val_df["risk_level"].tolist(),
                horizons=val_df["investment_horizon"].tolist(),
                actions=val_df["action"].tolist(),
                tokenizer=tokenizer,
            )
            eval_dataloader = DataLoader(
                val_dataset,
                sampler=SequentialSampler(val_dataset),
                batch_size=args.batch_size,
            )

        # Initialize model
        print("Initializing model")
        model = InvestmentTextClassifier(
            pretrained_model_name=tokenizer_name,
            risk_levels=settings.RISK_LEVELS,
            investment_horizons=settings.INVESTMENT_HORIZONS,
            actions=settings.ACTIONS,
        )

        # Train model
        print("Starting training")
        train_model(
            model=model,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            save_path=settings.MODEL_PATH,
            device=device,
        )

        print("Training completed")
    else:
        print(f"Error: Training data not found at {args.data_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the investment text classifier model"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=settings.TRAIN_DATA_PATH,
        help="Path to the training data",
    )
    parser.add_argument(
        "--validation-path",
        type=str,
        default=settings.VALIDATION_DATA_PATH,
        help="Path to the validation data",
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for training"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=2e-5, help="Learning rate"
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=0, help="Number of warmup steps"
    )

    args = parser.parse_args()
    main(args)
