import torch
from transformers.models.auto.tokenization_auto import AutoTokenizer

from app.models.pytorch.classifier import InvestmentTextClassifier


def classify_text(model, tokenizer, text):
    # Tokenize the text
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=512
    )

    # Remove token_type_ids if present as DistilBERT doesn't use them
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]

    # Make prediction
    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )

    # Get the predictions
    risk_level_idx = torch.argmax(outputs.risk_level_logits, dim=1).item()
    horizon_idx = torch.argmax(outputs.horizon_logits, dim=1).item()
    action_idx = torch.argmax(outputs.action_logits, dim=1).item()

    # Map indices to labels
    risk_level = model.risk_levels[risk_level_idx]
    horizon = model.investment_horizons[horizon_idx]
    action = model.actions[action_idx]

    # Print results
    print(f"Sample text: '{text}'")
    print(f"Model predictions:")
    print(f"Risk level: {risk_level}")
    print(f"Investment horizon: {horizon}")
    print(f"Action: {action}")
    print("-" * 80)


def test_model():
    # Initialize the model with the same parameters as in the original
    model = InvestmentTextClassifier(
        pretrained_model_name="distilbert-base-uncased",
        risk_levels=["high", "medium", "low"],
        investment_horizons=["short-term", "medium-term", "long-term"],
        actions=["buy", "sell", "hold", "ignore"],
    )

    # Set model to evaluation mode
    model.eval()

    # Load a pre-trained tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # Test different investment scenarios
    test_texts = [
        "This stock has shown strong growth over the past 6 months and analysts project continued upward momentum. Consider buying for medium-term gains.",
        "The company has experienced significant volatility recently due to pending litigation. Consider holding until the legal issues are resolved.",
        "This low-yield bond offers stable returns with minimal risk, ideal for conservative long-term investors approaching retirement.",
        "The startup's new product launch failed to meet expectations, and they're facing cash flow problems. Sell your position and cut losses.",
        "The market is showing signs of a potential recession. Consider diversifying your portfolio with defensive stocks.",
    ]

    for text in test_texts:
        classify_text(model, tokenizer, text)

    return True


if __name__ == "__main__":
    test_model()
