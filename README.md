# Investment Text Classifier

A machine learning model built with PyTorch that classifies investment-related text into actionable recommendations (buy, sell, hold, or ignore) based on specific client requirements and investment strategies (high risk, medium risk, low risk, short-term, etc.).

## Project Overview

This project aims to provide a text classification service that:

- Analyzes investment-related text content
- Classifies texts according to investment strategies and risk profiles
- Provides recommendations tailored to client preferences
- Serves the trained model via a RESTful API

## Technology Stack

- **Machine Learning**: PyTorch for model training and inference
- **API Framework**: FastAPI for serving predictions
- **Environment Management**: Poetry for dependency management
- **Containerization**: Docker for consistent development and deployment
- **CI/CD**: GitHub Actions for continuous integration and deployment

## Getting Started

### Prerequisites

- Python 3.8+
- Docker
- Poetry

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/investment-text-classifier.git
cd investment-text-classifier
```

2. Install dependencies using Poetry:

```bash
poetry install
```

3. Build and run with Docker:

```bash
docker build -t investment-text-classifier .
docker run -p 8000:8000 investment-text-classifier
```

## Development

### Project Structure

```
investment-text-classifier/
├── app/                    # FastAPI application
│   ├── api/                # API endpoints
│   ├── core/               # Core application configuration
│   ├── models/             # PyTorch models
│   └── services/           # Business logic
├── data/                   # Training and validation data
├── notebooks/              # Jupyter notebooks for exploration
├── tests/                  # Test suite
├── .github/                # GitHub Actions workflows
├── Dockerfile              # Docker configuration
├── poetry.lock             # Poetry lock file
├── pyproject.toml          # Poetry configuration
└── README.md               # Project documentation
```

### Training the Model

```bash
poetry run python -m app.models.train --data-path data/training --epochs 10
```

### Running Tests

```bash
poetry run pytest
```

### Local Development Server

```bash
poetry run uvicorn app.main:app --reload
```

## API Documentation

After starting the server, access the API documentation at:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Deployment

The project is configured for deployment to AWS using GitHub Actions.

## License

[MIT License](LICENSE)
