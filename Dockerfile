# Multi-stage Dockerfile for MoE Recommendation System
# This Dockerfile creates an optimized container for serving the MoE model

# Stage 1: Base image with Python and system dependencies
FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Stage 2: Dependencies installation
FROM base as dependencies

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 3: Model preparation (if needed)
FROM dependencies as model-prep

# Copy model training/preparation scripts
COPY scripts/prepare_model.py ./scripts/
COPY models/ ./models/

# This stage can be used to download or prepare the model
# For now, we'll create a placeholder model
RUN python -c "import torch; import sys; sys.path.append('/app'); from models.moe_transformer import MoEConfig, create_moe_model; config = MoEConfig(); model = create_moe_model(config, 1000000, 500000); torch.save(model.state_dict(), '/app/models/moe_model.pth'); print('Model prepared successfully')"

# Stage 4: Final application image
FROM base as app

# Set working directory
WORKDIR /app

# Copy Python dependencies from dependencies stage
COPY --from=dependencies /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# Copy application code
COPY api/ ./api/
COPY models/ ./models/
COPY scripts/ ./scripts/

# Copy prepared model
COPY --from=model-prep /app/models/moe_model.pth ./models/

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/config

# Set proper permissions
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose ports
EXPOSE 8000 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "api/main.py"]

# Stage 5: Development image (optional)
FROM app as development

# Install development dependencies
USER root
RUN pip install --no-cache-dir pytest pytest-asyncio black flake8 mypy

# Copy test files
COPY tests/ ./tests/

# Switch back to appuser
USER appuser

# Development command
CMD ["python", "-m", "pytest", "tests/", "-v"] 