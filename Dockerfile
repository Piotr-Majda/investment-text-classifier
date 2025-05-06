FROM python:3.9-slim as python-base

# Python setup
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    PYSETUP_PATH="/opt/pysetup" \
    VENV_PATH="/opt/pysetup/.venv" \
    PROJECT_PATH="/app"

ENV PATH="$POETRY_HOME/bin:$VENV_PATH/bin:$PATH"

# Builder stage
FROM python-base as builder

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl build-essential

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Set up working directory
WORKDIR $PYSETUP_PATH

# Copy project files
COPY poetry.lock pyproject.toml ./

# Install runtime dependencies
RUN poetry install --no-dev --no-root

# Final stage
FROM python-base as final

# Copy virtual environment from builder
COPY --from=builder $VENV_PATH $VENV_PATH

# Set up working directory
WORKDIR $PROJECT_PATH

# Copy project files
COPY . .

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 