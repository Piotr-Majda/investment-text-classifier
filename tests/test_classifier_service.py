import pytest
from unittest.mock import MagicMock

import torch

from app.core.config import settings
from app.models.prediction import TextClassificationResult
from app.services.classifier import ClassifierService


@pytest.fixture
def mock_torch_load(monkeypatch):
    """Mock torch.load function."""
    mock = MagicMock()
    monkeypatch.setattr("app.services.classifier.torch.load", mock)
    return mock


@pytest.fixture
def mock_tokenizer(monkeypatch):
    """Mock AutoTokenizer.from_pretrained function."""
    mock = MagicMock()
    monkeypatch.setattr("app.services.classifier.AutoTokenizer.from_pretrained", mock)
    return mock


@pytest.fixture
def dev_mode_service(monkeypatch):
    """Create a ClassifierService instance in development mode (no model exists)."""
    monkeypatch.setattr("app.services.classifier.os.path.exists", lambda _: False)
    return ClassifierService()


@pytest.fixture
def prod_mode_service(monkeypatch, mock_torch_load, mock_tokenizer):
    """Create a ClassifierService instance in production mode (model exists)."""
    # Set up for model existing
    monkeypatch.setattr("app.services.classifier.os.path.exists", lambda _: True)
    
    # Create mock model
    mock_model = MagicMock()
    mock_torch_load.return_value = mock_model
    
    # Create mock tokenizer instance
    mock_tokenizer_instance = MagicMock()
    mock_tokenizer.return_value = mock_tokenizer_instance
    
    return ClassifierService()


def test_init_dev_mode(dev_mode_service):
    """Test initialization in development mode (no model)."""
    # Check model and tokenizer are None
    assert dev_mode_service.model is None
    assert dev_mode_service.tokenizer is None
    
    # Check device is set
    assert dev_mode_service.device in ["cuda", "cpu"]
    
    # Check classification categories are set
    assert dev_mode_service.risk_levels == settings.RISK_LEVELS
    assert dev_mode_service.investment_horizons == settings.INVESTMENT_HORIZONS
    assert dev_mode_service.actions == settings.ACTIONS


def test_is_ready_dev_mode(dev_mode_service):
    """Test is_ready() method in development mode."""
    assert dev_mode_service.is_ready() is False


def test_predict_placeholder_high_risk(dev_mode_service):
    """Test predict method with high-risk keywords."""
    high_risk_text = "This investment is volatile and speculative."
    result = dev_mode_service.predict(high_risk_text)
    
    assert isinstance(result, TextClassificationResult)
    assert result.risk_level == "high"


def test_predict_placeholder_buy_action(dev_mode_service):
    """Test predict method with buy action keywords."""
    buy_text = "This is a great opportunity to buy these stocks."
    result = dev_mode_service.predict(buy_text)
    
    assert isinstance(result, TextClassificationResult)
    assert result.action == "buy"


def test_predict_placeholder_long_term(dev_mode_service):
    """Test predict method with long-term horizon keywords."""
    long_term_text = "Consider this for your retirement in future years."
    result = dev_mode_service.predict(long_term_text)
    
    assert isinstance(result, TextClassificationResult)
    assert result.investment_horizon == "long-term"


def test_init_with_model(prod_mode_service, mock_torch_load):
    """Test initialization with a model."""
    # Check model and tokenizer are set
    assert prod_mode_service.model is not None
    assert prod_mode_service.tokenizer is not None
    
    # Check model was put in eval mode
    mock_model = mock_torch_load.return_value
    mock_model.eval.assert_called_once() 