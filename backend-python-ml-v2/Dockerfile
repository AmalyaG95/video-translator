# Multi-stage Dockerfile following 2025 best practices
# See best-practices/cross-cutting/MODERN-2025-PRACTICES.md

# Stage 1: Build dependencies
FROM python:3.12-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV TERM=linux

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    pkg-config \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libavfilter-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Add /root/.local/bin to PATH for builder stage
ENV PATH=/root/.local/bin:$PATH

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime image
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV TERM=linux
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    fontconfig \
    fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

# Copy Python packages from builder to app directory (accessible by appuser)
COPY --from=builder /root/.local /app/.local

# Make sure scripts in .local are usable
ENV PATH=/app/.local/bin:$PATH

# Copy application code
COPY src/ ./src/
# Note: Proto files (translation_pb2.py, etc.) are generated at runtime by entrypoint.sh
# The translation.proto file is already included in src/proto/

# Create necessary directories
RUN mkdir -p /app/artifacts /app/temp_work /app/uploads /app/temp_work/checkpoints /app/temp_work/logs

# Set Python path to include local packages
ENV PYTHONPATH=/app/src:/app/.local/lib/python3.12/site-packages

# Create non-root user for security (2025 best practice)
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Copy entrypoint script with executable permissions (before switching user)
COPY --chmod=755 entrypoint.sh /app/entrypoint.sh

USER appuser

# Expose gRPC port
EXPOSE 50051

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import grpc; grpc.insecure_channel('localhost:50051')" || exit 1

# Default command
ENTRYPOINT ["/app/entrypoint.sh"]
