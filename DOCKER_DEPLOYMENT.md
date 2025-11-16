# Docker Deployment Guide

This guide explains how to deploy the Video Translation System using Docker.

## Prerequisites

- Docker Engine 20.10+ installed and running
- Docker Compose v2.0+ (or docker-compose v1.29+)
- At least 8GB RAM available for Docker
- 20GB+ free disk space

## Quick Start

### 1. Deploy with Script (Recommended)

```bash
# Make script executable (if not already)
chmod +x deploy.sh

# Start all services
./deploy.sh up --build

# View logs
./deploy.sh logs

# Stop services
./deploy.sh down
```

### 2. Manual Deployment

```bash
# Build all images
docker-compose build

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Service Architecture

The system consists of three services:

1. **Frontend** (Next.js)

   - Port: `3000`
   - URL: http://localhost:3000
   - Built with Node.js 20 Alpine

2. **NestJS API** (Backend Gateway)

   - Port: `3001`
   - URL: http://localhost:3001
   - Communicates with Python ML service via gRPC

3. **Python ML Service** (gRPC Microservice)
   - Port: `50051`
   - Handles video processing (STT, Translation, TTS)
   - Resource limits: 8GB RAM, 4 CPUs

## Configuration

### Environment Variables

Create a `.env` file in the project root (optional):

```env
# Frontend API URL (for browser access)
NEXT_PUBLIC_API_URL=http://localhost:3001

# API Configuration
PORT=3001
NODE_ENV=production

# Python ML Service
GRPC_ML_SERVICE_URL=python-ml:50051
```

If `.env` is not provided, defaults will be used.

### Volume Mounts

The following directories are mounted as volumes:

- `./uploads` → Video upload storage (shared between NestJS and Python ML)
- `./artifacts` → Processing artifacts (shared between services)
- `./backend-python-ml/src/config` → ML service configuration
- `temp-work` → Temporary work directory (Docker volume)

**Important:** Ensure these directories exist and have proper permissions:

```bash
mkdir -p uploads artifacts temp_work backend-python-ml/temp_work
chmod -R 755 uploads artifacts temp_work
```

## Deployment Commands

### Using deploy.sh script:

```bash
# Build images
./deploy.sh build

# Start services (with rebuild)
./deploy.sh up --build

# Start services (no rebuild)
./deploy.sh up

# Stop services
./deploy.sh down

# Restart services
./deploy.sh restart

# View logs (all services)
./deploy.sh logs

# View logs (specific service)
./deploy.sh logs frontend
./deploy.sh logs nestjs-api
./deploy.sh logs python-ml

# Check service status
./deploy.sh status

# Clean up everything (removes containers, volumes, images)
./deploy.sh clean
```

### Using docker-compose directly:

```bash
# Build
docker-compose build

# Start
docker-compose up -d

# Stop
docker-compose down

# View logs
docker-compose logs -f [service-name]

# Restart a service
docker-compose restart [service-name]

# View status
docker-compose ps

# Remove volumes too
docker-compose down -v
```

## Health Checks

The Python ML service includes a health check that verifies the gRPC server is running:

- **Interval:** 30 seconds
- **Timeout:** 10 seconds
- **Retries:** 3

Check service health:

```bash
docker-compose ps
```

Healthy services will show `Up (healthy)` status.

## Troubleshooting

### Services won't start

1. **Check Docker is running:**

   ```bash
   docker info
   ```

2. **Check ports are available:**

   ```bash
   # Check if ports are in use
   lsof -i :3000
   lsof -i :3001
   lsof -i :50051
   ```

3. **Check logs:**
   ```bash
   docker-compose logs [service-name]
   ```

### Frontend can't connect to API

- Ensure `NEXT_PUBLIC_API_URL` is set to `http://localhost:3001` (not the Docker service name)
- The browser runs on the host machine, not inside Docker, so it needs `localhost`

### Python ML service crashes

1. **Check memory limits:**

   ```bash
   docker stats
   ```

   Ensure at least 4GB RAM is available.

2. **Check logs:**

   ```bash
   docker-compose logs python-ml
   ```

3. **Check proto file generation:**
   The entrypoint script should generate proto files automatically. Check logs for errors.

### Permission errors

If you see permission errors with volumes:

```bash
# Fix permissions
sudo chown -R $USER:$USER uploads artifacts temp_work
chmod -R 755 uploads artifacts temp_work
```

Or run Docker with proper user mapping (see Docker documentation).

## Production Considerations

### Security

1. **Use secrets management:**

   - Don't commit `.env` files
   - Use Docker secrets or external secret management

2. **Network isolation:**

   - Services communicate via internal Docker network
   - Only expose necessary ports (3000, 3001)

3. **Resource limits:**
   - Adjust resource limits in `docker-compose.yml` based on your server capacity
   - Monitor resource usage: `docker stats`

### Performance

1. **Resource allocation:**

   - Python ML service: 4-8GB RAM, 2-4 CPUs recommended
   - Adjust in `docker-compose.yml` under `deploy.resources`

2. **Volume performance:**

   - Consider using named volumes for better performance
   - For production, use dedicated storage for `uploads` and `artifacts`

3. **Build optimization:**
   - Use `.dockerignore` to exclude unnecessary files
   - Leverage Docker layer caching for faster rebuilds

### Monitoring

1. **View resource usage:**

   ```bash
   docker stats
   ```

2. **Service health:**

   ```bash
   docker-compose ps
   ```

3. **Application logs:**
   ```bash
   docker-compose logs -f --tail=100
   ```

## Updating Services

To update after code changes:

```bash
# Rebuild and restart
./deploy.sh up --build

# Or manually
docker-compose build
docker-compose up -d
```

## Backup and Restore

### Backup volumes:

```bash
# Backup uploads
docker run --rm -v translate-v_uploads:/data -v $(pwd):/backup alpine tar czf /backup/uploads-backup.tar.gz /data

# Backup artifacts
docker run --rm -v translate-v_artifacts:/data -v $(pwd):/backup alpine tar czf /backup/artifacts-backup.tar.gz /data
```

### Restore:

```bash
# Restore uploads
docker run --rm -v translate-v_uploads:/data -v $(pwd):/backup alpine tar xzf /backup/uploads-backup.tar.gz -C /

# Restore artifacts
docker run --rm -v translate-v_artifacts:/data -v $(pwd):/backup alpine tar xzf /backup/artifacts-backup.tar.gz -C /
```

## Clean Up

### Remove everything:

```bash
./deploy.sh clean
```

Or manually:

```bash
# Stop and remove containers, networks, volumes, images
docker-compose down -v --rmi all

# Remove any orphaned volumes
docker volume prune
```

## Support

For issues or questions:

1. Check logs: `./deploy.sh logs`
2. Check service status: `./deploy.sh status`
3. Review this documentation
4. Check Docker and Docker Compose versions

