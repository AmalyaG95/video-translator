# Video Translator - Multi-stage Dockerfile
# Optimized build with security hardening

# Stage 1: Python Dependencies Builder
FROM python:3.11-slim AS python-builder

# Set build arguments
ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies for building Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    make \
    cmake \
    pkg-config \
    libffi-dev \
    libssl-dev \
    libjpeg-dev \
    libpng-dev \
    libfreetype6-dev \
    libxft-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    libhdf5-dev \
    libblas-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy requirements first for better caching
COPY backend/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Frontend Builder
FROM node:18-alpine AS frontend-builder

# Set working directory
WORKDIR /app

# Copy package files
COPY frontend/package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy frontend source
COPY frontend/ .

# Build the application
RUN npm run build

# Stage 3: Runtime Image
FROM python:3.11-slim

# Security: Create non-root user
RUN groupadd -r translator && useradd -r -g translator -s /bin/false translator

# Set build arguments
ARG DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgomp1 \
    libgfortran5 \
    libopenblas0 \
    liblapack3 \
    libhdf5-103 \
    libjpeg62-turbo \
    libpng16-16 \
    libfreetype6 \
    libxft2 \
    libxml2 \
    libxslt1.1 \
    zlib1g \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy Python dependencies from builder
COPY --from=python-builder /root/.local /home/translator/.local

# Copy frontend build from builder
COPY --from=frontend-builder /app/.next /app/frontend/.next
COPY --from=frontend-builder /app/public /app/frontend/public
COPY --from=frontend-builder /app/package.json /app/frontend/package.json

# Copy backend source
COPY backend/ /app/backend/
COPY run.sh /app/
COPY Makefile /app/

# Set working directory
WORKDIR /app

# Create necessary directories
RUN mkdir -p artifacts temp_work backend/uploads backend/temp \
    && chown -R translator:translator /app

# Set environment variables
ENV PATH=/home/translator/.local/bin:$PATH
ENV PYTHONPATH=/app/backend
ENV PYTHONUNBUFFERED=1
ENV NODE_ENV=production

# Expose ports
EXPOSE 8000 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Security: Switch to non-root user
USER translator

# Resource limits (set via docker run or docker-compose)
# --memory=8g --cpus=4

# Default command
CMD ["./run.sh"]

# Labels for metadata
LABEL maintainer="Video Translator"
LABEL version="1.0.0"
LABEL description="Video Translator - Production Build"
LABEL org.opencontainers.image.title="Video Translator"
LABEL org.opencontainers.image.description="End-to-end video translation with exact duration preservation"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.created="2024-01-15T00:00:00Z"