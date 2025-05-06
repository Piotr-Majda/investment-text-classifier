from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.models.auto.modeling_auto import AutoModel


@dataclass
class InvestmentClassifierOutput:
    """Output of the investment text classifier model."""

    risk_level_logits: torch.Tensor
    horizon_logits: torch.Tensor
    action_logits: torch.Tensor
    loss: Optional[torch.Tensor] = None


class InvestmentTextClassifier(nn.Module):
    """
    Multi-task text classifier for investment texts.

    Classifies text into:
    1. Risk level (high, medium, low)
    2. Investment horizon (short-term, medium-term, long-term)
    3. Action (buy, sell, hold, ignore)
    """

    def __init__(
        self,
        pretrained_model_name: str = "distilbert-base-uncased",
        risk_levels: List[str] = ["high", "medium", "low"],
        investment_horizons: List[str] = ["short-term", "medium-term", "long-term"],
        actions: List[str] = ["buy", "sell", "hold", "ignore"],
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        # Base transformer model
        self.transformer = AutoModel.from_pretrained(pretrained_model_name)
        self.hidden_size = self.transformer.config.hidden_size

        # Shared layers
        self.dropout = nn.Dropout(dropout_rate)

        # Task-specific heads
        self.risk_level_classifier = nn.Linear(self.hidden_size, len(risk_levels))
        self.horizon_classifier = nn.Linear(self.hidden_size, len(investment_horizons))
        self.action_classifier = nn.Linear(self.hidden_size, len(actions))

        # Store output classes
        self.risk_levels = risk_levels
        self.investment_horizons = investment_horizons
        self.actions = actions

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        risk_level_labels: Optional[torch.Tensor] = None,
        horizon_labels: Optional[torch.Tensor] = None,
        action_labels: Optional[torch.Tensor] = None,
    ) -> InvestmentClassifierOutput:
        """Forward pass for the classifier."""
        # Get transformer outputs
        transformer_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            transformer_inputs["token_type_ids"] = token_type_ids

        outputs = self.transformer(**transformer_inputs)

        # Get [CLS] token representation (sentence embedding)
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)

        # Task-specific predictions
        risk_level_logits = self.risk_level_classifier(pooled_output)
        horizon_logits = self.horizon_classifier(pooled_output)
        action_logits = self.action_classifier(pooled_output)

        # Calculate loss if labels are provided
        loss = None
        if (
            risk_level_labels is not None
            and horizon_labels is not None
            and action_labels is not None
        ):
            loss_fct = nn.CrossEntropyLoss()
            risk_level_loss = loss_fct(risk_level_logits, risk_level_labels)
            horizon_loss = loss_fct(horizon_logits, horizon_labels)
            action_loss = loss_fct(action_logits, action_labels)

            # Combined loss (can adjust weights if needed)
            loss = risk_level_loss + horizon_loss + action_loss

        return InvestmentClassifierOutput(
            risk_level_logits=risk_level_logits,
            horizon_logits=horizon_logits,
            action_logits=action_logits,
            loss=loss,
        )
