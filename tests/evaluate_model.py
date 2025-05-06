import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers.models.auto.tokenization_auto import AutoTokenizer

from app.core.config import settings


def evaluate_trained_model():
    print("Loading trained model...")
    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(settings.MODEL_PATH, map_location=device)
    model.eval()  # Set model to evaluation mode

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_TOKENIZER)

    # Load test data
    print("Loading test data...")
    test_data = pd.read_csv("data/test/test_data.csv")

    # Prepare lists to store results
    texts = test_data["text"].tolist()
    true_risk_levels = test_data["risk_level"].tolist()
    true_horizons = test_data["investment_horizon"].tolist()
    true_actions = test_data["action"].tolist()

    # Risk level, horizon, and action mapping
    risk_level_map = {i: label for i, label in enumerate(settings.RISK_LEVELS)}
    horizon_map = {i: label for i, label in enumerate(settings.INVESTMENT_HORIZONS)}
    action_map = {i: label for i, label in enumerate(settings.ACTIONS)}

    # Convert labels to indices
    risk_level_idx_map = {label: i for i, label in enumerate(settings.RISK_LEVELS)}
    horizon_idx_map = {label: i for i, label in enumerate(settings.INVESTMENT_HORIZONS)}
    action_idx_map = {label: i for i, label in enumerate(settings.ACTIONS)}

    true_risk_indices = [risk_level_idx_map[level] for level in true_risk_levels]
    true_horizon_indices = [horizon_idx_map[horizon] for horizon in true_horizons]
    true_action_indices = [action_idx_map[action] for action in true_actions]

    # Prediction lists
    pred_risk_levels = []
    pred_horizons = []
    pred_actions = []

    pred_risk_indices = []
    pred_horizon_indices = []
    pred_action_indices = []

    # Process each text
    print("\nEvaluating model on test data...")
    print("-" * 80)
    for i, text in enumerate(texts):
        # Tokenize
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=512
        )
        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]

        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)

        # Get prediction indices
        risk_level_idx = torch.argmax(outputs.risk_level_logits, dim=1).item()
        horizon_idx = torch.argmax(outputs.horizon_logits, dim=1).item()
        action_idx = torch.argmax(outputs.action_logits, dim=1).item()

        # Map indices to labels
        risk_level = risk_level_map[risk_level_idx]
        horizon = horizon_map[horizon_idx]
        action = action_map[action_idx]

        # Save predictions
        pred_risk_levels.append(risk_level)
        pred_horizons.append(horizon)
        pred_actions.append(action)

        pred_risk_indices.append(risk_level_idx)
        pred_horizon_indices.append(horizon_idx)
        pred_action_indices.append(action_idx)

        # Print individual prediction
        print(f"Example {i+1}:")
        print(f"Text: {text[:100]}...")
        print(f"Prediction: Risk={risk_level}, Horizon={horizon}, Action={action}")
        print(
            f"Actual: Risk={true_risk_levels[i]}, Horizon={true_horizons[i]}, Action={true_actions[i]}"
        )
        print("-" * 80)

    # Calculate metrics
    print("\nOverall Performance Metrics:")
    print("-" * 40)

    # Risk level metrics
    risk_accuracy = accuracy_score(true_risk_indices, pred_risk_indices)
    risk_f1 = f1_score(true_risk_indices, pred_risk_indices, average="weighted")
    print(f"Risk Level Accuracy: {risk_accuracy:.4f}")
    print(f"Risk Level F1 Score: {risk_f1:.4f}")
    print("\nRisk Level Classification Report:")
    print(
        classification_report(
            true_risk_indices, pred_risk_indices, target_names=settings.RISK_LEVELS
        )
    )

    # Horizon metrics
    horizon_accuracy = accuracy_score(true_horizon_indices, pred_horizon_indices)
    horizon_f1 = f1_score(
        true_horizon_indices, pred_horizon_indices, average="weighted"
    )
    print(f"Investment Horizon Accuracy: {horizon_accuracy:.4f}")
    print(f"Investment Horizon F1 Score: {horizon_f1:.4f}")
    print("\nInvestment Horizon Classification Report:")
    print(
        classification_report(
            true_horizon_indices,
            pred_horizon_indices,
            target_names=settings.INVESTMENT_HORIZONS,
        )
    )

    # Action metrics
    action_accuracy = accuracy_score(true_action_indices, pred_action_indices)
    action_f1 = f1_score(true_action_indices, pred_action_indices, average="weighted")
    print(f"Action Accuracy: {action_accuracy:.4f}")
    print(f"Action F1 Score: {action_f1:.4f}")
    print("\nAction Classification Report:")
    print(
        classification_report(
            true_action_indices, pred_action_indices, target_names=settings.ACTIONS
        )
    )

    # Calculate overall accuracy (all three predictions correct)
    correct = 0
    for i in range(len(texts)):
        if (
            pred_risk_levels[i] == true_risk_levels[i]
            and pred_horizons[i] == true_horizons[i]
            and pred_actions[i] == true_actions[i]
        ):
            correct += 1
    overall_accuracy = correct / len(texts)
    print(f"\nOverall Accuracy (all three correct): {overall_accuracy:.4f}")


if __name__ == "__main__":
    evaluate_trained_model()
