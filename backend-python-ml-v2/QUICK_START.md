# Quick Start Guide

## Prerequisites

- Docker and Docker Compose
- Python 3.11+ (for local development)
- FFmpeg installed (for local development)

## Quick Start with Docker

```bash
# Build the service
cd backend-python-ml-v2
docker build -t video-translation-ml-v2 .

# Or use docker-compose
cd ..
docker-compose -f docker-compose.v2.yml up --build
```

## Local Development

```bash
# Install dependencies
pip install -r requirements-dev.txt

# Generate proto files
./generate_proto.sh

# Run service
python3 -m src.main
```

## Testing

```bash
# Run all tests
pytest

# Run specific test category
pytest tests/unit/
pytest tests/integration/
pytest tests/e2e/

# With coverage
pytest --cov=src --cov-report=html
```

## Integration with NestJS

The service is compatible with the existing NestJS API. To use it:

1. Update `GRPC_ML_SERVICE_URL` in NestJS environment to `python-ml-v2:50051`
2. Restart NestJS service
3. The new service will handle all translation requests

## Configuration

Edit `src/config/config.yaml` or set environment variables:

```bash
export MAX_MEMORY_GB=8
export GRPC_PORT=50051
export ENABLE_TRACING=true
```

## Troubleshooting

### Proto Import Errors
- Ensure proto files are generated: `./generate_proto.sh`
- Check that `grpc-tools` is installed

### Memory Issues
- Reduce `MAX_MEMORY_GB` if needed
- Model manager will auto-unload unused models

### Service Won't Start
- Check logs: `docker-compose logs python-ml-v2`
- Verify all dependencies are installed
- Check configuration file syntax

## Next Steps

1. Review `IMPLEMENTATION_STATUS.md` for completion status
2. Review `SUMMARY.md` for architecture overview
3. Review `DEPLOYMENT.md` for deployment details
4. Follow `best-practices/README.md` for implementation patterns



