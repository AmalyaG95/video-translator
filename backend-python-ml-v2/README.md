# Video Translation ML Service v2

A professional-grade Python ML service for video translation, built from scratch following industry best practices.

## 🚀 Quick Start

```bash
# Build and run with Docker
docker build -t video-translation-ml-v2 .
docker run -p 50051:50051 video-translation-ml-v2

# Or use docker-compose
cd ..
docker-compose -f docker-compose.v2.yml up --build
```

## 📋 Overview

This service provides a complete video translation pipeline:

1. **Model Initialization** - Lazy loading of ML models
2. **Audio Extraction** - Extract audio from video
3. **Speech-to-Text** - Transcribe audio using Whisper
4. **Translation** - Translate text using Helsinki-NLP
5. **Text-to-Speech** - Generate speech using Edge-TTS
6. **Audio Synchronization** - Sync audio with original timing
7. **Subtitle Generation** - Create SRT/WebVTT subtitles
8. **Video Combination** - Combine video with translated audio

## 🏗️ Architecture

- **Microservices Design** - Separate from NestJS API
- **Async-First** - All operations are asynchronous
- **Type-Safe** - Pydantic for configuration and validation
- **Resource Management** - Automatic memory and CPU monitoring
- **State Persistence** - Checkpointing for resume capability
- **Quality Validation** - Quality checks at each stage
- **Observability** - OpenTelemetry tracing support

## 📁 Project Structure

```
backend-python-ml-v2/
├── src/
│   ├── config/          # Configuration management
│   ├── logging/         # Structured logging
│   ├── utils/           # Utility functions
│   ├── core/            # Core infrastructure
│   ├── pipeline/        # Translation pipeline
│   ├── services/        # gRPC service layer
│   ├── observability/   # Tracing and monitoring
│   ├── proto/           # gRPC protocol definitions
│   └── main.py          # Entry point
├── tests/               # Test suite
├── best-practices/      # Best practices documentation
└── docs/                # Additional documentation
```

## 🔧 Configuration

Edit `src/config/config.yaml` or set environment variables:

```bash
export MAX_MEMORY_GB=8
export GRPC_PORT=50051
export ENABLE_TRACING=true
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test category
pytest tests/unit/
pytest tests/integration/

# With coverage
pytest --cov=src --cov-report=html
```

## 📚 Documentation

- **[QUICK_START.md](QUICK_START.md)** - Quick start guide
- **[SUMMARY.md](SUMMARY.md)** - Architecture summary
- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Deployment guide
- **[CHECKLIST.md](CHECKLIST.md)** - Pre-deployment checklist
- **[best-practices/](../best-practices/)** - Comprehensive best practices

## 🔌 Integration

The service is compatible with the existing NestJS API:

1. Update `GRPC_ML_SERVICE_URL` in NestJS to `python-ml-v2:50051`
2. Restart NestJS service
3. The new service handles all translation requests

## 🛠️ Development

```bash
# Install dependencies
pip install -r requirements-dev.txt

# Generate proto files
./generate_proto.sh

# Run service locally
python3 -m src.main

# Format code
black src/ tests/
ruff format src/ tests/

# Lint
ruff check src/
mypy src/
```

## 📊 Features

- ✅ **8 Pipeline Stages** - Complete translation pipeline
- ✅ **Resource Management** - Automatic memory/CPU monitoring
- ✅ **Checkpointing** - Resume from any stage
- ✅ **Quality Validation** - Quality checks throughout
- ✅ **Error Handling** - Retry with exponential backoff
- ✅ **Progress Streaming** - Real-time progress updates
- ✅ **Cancellation** - Support for request cancellation
- ✅ **Background Audio** - Preserves original audio
- ✅ **Subtitle Generation** - SRT and WebVTT support
- ✅ **Observability** - OpenTelemetry tracing

## 🐳 Docker

The service runs in Docker with:
- Multi-stage build for smaller images
- Automatic proto generation
- Health checks
- Resource limits
- Volume mounts for data persistence

## 📝 License

[Your License Here]

## 🤝 Contributing

[Your Contributing Guidelines Here]

## 📞 Support

For issues and questions, see:
- [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)
- [FINAL_NOTES.md](FINAL_NOTES.md)
- [best-practices/README.md](../best-practices/README.md)
