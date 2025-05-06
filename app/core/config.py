import os
from typing import List, Optional, Union

from pydantic import AnyHttpUrl, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Investment Text Classifier"
    
    # CORS settings
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:8000",
        "http://localhost:3000",  # Frontend development server
    ]

    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    # Model settings
    MODEL_PATH: str = "models/classifier.pt"
    MODEL_TOKENIZER: str = "models/tokenizer"
    
    # Training settings
    TRAIN_DATA_PATH: str = "data/training"
    VALIDATION_DATA_PATH: Optional[str] = "data/validation"
    TEST_DATA_PATH: Optional[str] = "data/test"
    
    # Investment strategy classifications
    RISK_LEVELS: List[str] = ["high", "medium", "low"]
    INVESTMENT_HORIZONS: List[str] = ["short-term", "medium-term", "long-term"]
    ACTIONS: List[str] = ["buy", "sell", "hold", "ignore"]

    class Config:
        case_sensitive = True
        env_file = ".env"


settings = Settings() 