# Getting Started with ML Service v2

Welcome! This guide will help you get the new ML service up and running.

## Prerequisites

- Docker and Docker Compose installed
- Python 3.11+ (for local development)
- FFmpeg (for local development)
- ~10GB free disk space (for models)

## Quick Start (5 minutes)

### Step 1: Build the Service

```bash
cd backend-python-ml-v2
docker build -t video-translation-ml-v2 .
```

### Step 2: Start with Docker Compose

```bash
cd ..
docker-compose -f docker-compose.v2.yml up --build
```

The service will:
- Generate proto files automatically
- Download models on first use
- Start gRPC server on port 50051

### Step 3: Verify It's Working

```bash
# Check service logs
docker-compose -f docker-compose.v2.yml logs python-ml-v2

# Should see:
# - "Proto files generated"
# - "gRPC server started on port 50051"
# - "Service ready to accept requests"
```

## Local Development

### Setup

```bash
# Install dependencies
pip install -r requirements-dev.txt

# Generate proto files
./generate_proto.sh

# Verify imports
python3 verify_imports.py

# Test startup
python3 test_startup.py
```

### Run Service Locally

```bash
# Start the service
python3 -m src.main

# Service will listen on localhost:50051
```

## Integration with NestJS

The service is already configured in `docker-compose.v2.yml`:

1. **NestJS** automatically connects to `python-ml-v2:50051`
2. **No code changes** needed in NestJS
3. **Same gRPC interface** as old service

To use with existing setup:

```bash
# Update NestJS environment
export GRPC_ML_SERVICE_URL=python-ml-v2:50051

# Restart NestJS
docker-compose restart nestjs-api
```

## First Translation

1. **Upload a video** through the frontend
2. **Start translation** (select source/target languages)
3. **Monitor progress** in logs:
   ```bash
   docker-compose -f docker-compose.v2.yml logs -f python-ml-v2
   ```

## What Happens During Translation

1. **Model Initialization** - Loads Whisper, translation, TTS models
2. **Audio Extraction** - Extracts audio from video
3. **Speech-to-Text** - Transcribes audio
4. **Translation** - Translates text
5. **Text-to-Speech** - Generates translated speech
6. **Audio Sync** - Synchronizes with original timing
7. **Subtitle Generation** - Creates SRT/WebVTT files
8. **Video Combination** - Combines video with translated audio

## Troubleshooting

### Service Won't Start

```bash
# Check logs
docker-compose -f docker-compose.v2.yml logs python-ml-v2

# Verify proto files
docker-compose -f docker-compose.v2.yml exec python-ml-v2 \
  ls -la src/proto/*_pb2.py

# Test imports
docker-compose -f docker-compose.v2.yml exec python-ml-v2 \
  python3 verify_imports.py
```

### Out of Memory

```bash
# Check Docker resource limits
docker stats python-ml-v2

# Increase memory in docker-compose.v2.yml:
#   deploy:
#     resources:
#       limits:
#         memory: 8G  # Increase if needed
```

### Models Not Downloading

- First run downloads models (can take 10-30 minutes)
- Check internet connection
- Verify disk space
- Check logs for download progress

## Next Steps

1. **Read Documentation**
   - [README.md](README.md) - Service overview
   - [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) - Integration details
   - [DEPLOYMENT.md](DEPLOYMENT.md) - Production deployment

2. **Explore Code**
   - `src/pipeline/` - Pipeline stages
   - `src/core/` - Core infrastructure
   - `src/services/` - gRPC service

3. **Run Tests**
   ```bash
   pytest tests/
   ```

4. **Review Best Practices**
   - `best-practices/` - Comprehensive documentation

## Support

- **Documentation**: See [INDEX.md](INDEX.md)
- **Issues**: Check logs and verification scripts
- **Best Practices**: Review `best-practices/` directory

## Success!

If you see "Service ready to accept requests" in the logs, you're all set! 🎉

The service is now ready to process video translations.



