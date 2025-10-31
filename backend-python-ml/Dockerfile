FROM python:3.11-slim

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TERM=linux

# Install system dependencies including build tools for PyAV
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    pkg-config \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libavfilter-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/

# Copy entrypoint script
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Create necessary directories
RUN mkdir -p /app/artifacts /app/temp_work /app/uploads

# Set Python path
ENV PYTHONPATH=/app/src

# Expose gRPC port
EXPOSE 50051

# Use entrypoint script that generates proto files and fixes imports at startup
ENTRYPOINT ["/app/entrypoint.sh"]

