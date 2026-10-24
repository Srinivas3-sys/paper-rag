# Multi-stage Dockerfile for paper-rag production service
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create non-root user for security best practices
RUN groupadd -r appuser && useradd -r -g appuser -d /app appuser

# Install python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application source code and scripts
COPY src/ /app/src/
COPY Makefile /app/
COPY README.md /app/

# Create data and indices directories with appropriate permissions
RUN mkdir -p /app/data /app/indices /app/results && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch FastAPI service with Uvicorn
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
