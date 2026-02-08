# DrishtiChokepointAgent Dockerfile
# ==================================
# Cloud Run compatible container definition.
#
# Build: docker build -t drishti-agent .
# Run:   docker run -p 8001:8001 drishti-agent

FROM python:3.11-slim

# Metadata
LABEL maintainer="Drishti Project"
LABEL description="Physics-grounded crowd safety reasoning agent"
LABEL version="0.1.0"

# Security: Run as non-root user
RUN useradd --create-home --shell /bin/bash appuser

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config.yaml .
COPY data/ ./data/

# Change ownership to non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port (Cloud Run uses PORT env variable)
EXPOSE 8001

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8001}/health || exit 1

# Run the application
# Cloud Run sets PORT environment variable
CMD ["sh", "-c", "uvicorn src.drishti_agent.main:app --host 0.0.0.0 --port ${PORT:-8001}"]
