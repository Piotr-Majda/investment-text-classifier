import os
from functools import lru_cache
from typing import Dict, List, Optional

import torch
from transformers import AutoTokenizer

from app.core.config import settings
from app.models.prediction import TextClassificationResult
from app.models.pytorch.classifier import InvestmentTextClassifier


class ClassifierService:
    def __init__(self) -> None:
        self.model: Optional[InvestmentTextClassifier] = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.risk_levels = settings.RISK_LEVELS
        self.investment_horizons = settings.INVESTMENT_HORIZONS
        self.actions = settings.ACTIONS
        self._load_model()

    def _load_model(self) -> None:
        """Load the PyTorch model and tokenizer."""
        try:
            if os.path.exists(settings.MODEL_PATH):
                self.model = torch.load(settings.MODEL_PATH, map_location=self.device)
                self.model.eval()
                self.tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_TOKENIZER)
            else:
                # For development, if model doesn't exist yet, we'll use placeholders
                self.model = None
                self.tokenizer = None
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
            self.tokenizer = None

    def is_ready(self) -> bool:
        """Check if the model is loaded and ready for inference."""
        return self.model is not None and self.tokenizer is not None

    def predict(self, text: str) -> TextClassificationResult:
        """
        Make predictions on the input text.
        
        For development, if no model is loaded, return placeholder predictions.
        In production, this should make actual predictions using the loaded model.
        """
        if not self.is_ready():
            # Return placeholder predictions for development
            return self._get_placeholder_prediction(text)
        
        # Tokenize the input text
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        # Make predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Process outputs to get the classifications and confidence scores
        risk_level_scores = torch.softmax(outputs.risk_level_logits, dim=1)[0].tolist()
        horizon_scores = torch.softmax(outputs.horizon_logits, dim=1)[0].tolist()
        action_scores = torch.softmax(outputs.action_logits, dim=1)[0].tolist()
        
        # Get predicted classes
        risk_level_idx = torch.argmax(outputs.risk_level_logits, dim=1).item()
        horizon_idx = torch.argmax(outputs.horizon_logits, dim=1).item()
        action_idx = torch.argmax(outputs.action_logits, dim=1).item()
        
        # Create confidence score dicts
        risk_level_dict = {level: score for level, score in zip(self.risk_levels, risk_level_scores)}
        horizon_dict = {horizon: score for horizon, score in zip(self.investment_horizons, horizon_scores)}
        action_dict = {action: score for action, score in zip(self.actions, action_scores)}
        
        return TextClassificationResult(
            risk_level=self.risk_levels[risk_level_idx],
            investment_horizon=self.investment_horizons[horizon_idx],
            action=self.actions[action_idx],
            confidence_scores={
                "risk_level": risk_level_dict,
                "investment_horizon": horizon_dict,
                "action": action_dict
            }
        )
        
    def _get_placeholder_prediction(self, text: str) -> TextClassificationResult:
        """Return placeholder predictions for development when no model is loaded."""
        # Simple rule-based placeholders based on keywords in the text
        text_lower = text.lower()
        
        # Risk level determination based on keywords
        risk_level = "medium"  # Default
        if any(word in text_lower for word in ["volatile", "high risk", "aggressive", "speculative"]):
            risk_level = "high"
        elif any(word in text_lower for word in ["safe", "conservative", "low risk", "stable"]):
            risk_level = "low"
            
        # Investment horizon determination based on keywords
        horizon = "medium-term"  # Default
        if any(word in text_lower for word in ["long term", "years", "decade", "future"]):
            horizon = "long-term"
        elif any(word in text_lower for word in ["short term", "quick", "immediate", "day trade"]):
            horizon = "short-term"
            
        # Action determination based on keywords
        action = "hold"  # Default
        if any(word in text_lower for word in ["buy", "purchase", "acquire", "opportunity"]):
            action = "buy"
        elif any(word in text_lower for word in ["sell", "exit", "bearish", "overvalued"]):
            action = "sell"
        elif any(word in text_lower for word in ["avoid", "pass", "uncertain"]):
            action = "ignore"
        
        # Generate placeholder confidence scores
        return TextClassificationResult(
            risk_level=risk_level,
            investment_horizon=horizon,
            action=action,
            confidence_scores={
                "risk_level": {level: 0.8 if level == risk_level else 0.1 for level in self.risk_levels},
                "investment_horizon": {h: 0.8 if h == horizon else 0.1 for h in self.investment_horizons},
                "action": {a: 0.8 if a == action else 0.06667 for a in self.actions}
            }
        )


@lru_cache()
def get_classifier_service() -> ClassifierService:
    """Singleton pattern for the classifier service."""
    return ClassifierService() 