from typing import List, Optional

from pydantic import BaseModel, Field


class TextClassificationInput(BaseModel):
    text: str = Field(..., description="Investment-related text to classify")
    
    class Config:
        schema_extra = {
            "example": {
                "text": "Tech stocks are showing strong growth potential with the rise of AI. "
                "This sector could provide significant returns over the next 5 years, "
                "though there might be short-term volatility."
            }
        }


class TextClassificationResult(BaseModel):
    risk_level: str = Field(..., description="Classified risk level (high, medium, low)")
    investment_horizon: str = Field(
        ..., description="Classified investment horizon (short-term, medium-term, long-term)"
    )
    action: str = Field(..., description="Recommended action (buy, sell, hold, ignore)")
    confidence_scores: dict = Field(
        ..., description="Confidence scores for each classification category"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "risk_level": "medium",
                "investment_horizon": "long-term",
                "action": "buy",
                "confidence_scores": {
                    "risk_level": {
                        "high": 0.2,
                        "medium": 0.7,
                        "low": 0.1
                    },
                    "investment_horizon": {
                        "short-term": 0.1,
                        "medium-term": 0.2,
                        "long-term": 0.7
                    },
                    "action": {
                        "buy": 0.8,
                        "sell": 0.1,
                        "hold": 0.05,
                        "ignore": 0.05
                    }
                }
            }
        } 