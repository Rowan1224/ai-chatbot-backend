# Multi-stage build for smaller image size
# Stage 1: Builder
FROM python:3.11-slim as builder

# Install uv for faster dependency installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies in a virtual environment
RUN uv sync --frozen --no-dev

# Stage 2: Runtime
FROM python:3.11-slim

# Install security updates and clean up
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app && \
    chown -R appuser:appuser /app

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

# Copy application code
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser pyproject.toml ./

# Switch to non-root user
USER appuser

# Add virtual environment to PATH
ENV PATH="/app/.venv/bin:$PATH"

# Expose port
EXPOSE 8000

# Health check — uses stdlib urllib.request; no third-party package needed
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=5)" || exit 1

# Run with Gunicorn + Uvicorn workers for production
# --workers: Number of worker processes (2-4 x CPU cores recommended)
# --worker-class: Use Uvicorn's async worker
# --bind: Listen on all interfaces
# --timeout: Worker timeout (important for LLM calls)
# --graceful-timeout: Time to wait for graceful shutdown
# --keep-alive: Keep connections alive
CMD ["gunicorn", "src.api.main:app", \
     "--workers", "1", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120", \
     "--graceful-timeout", "30", \
     "--keep-alive", "5", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]


