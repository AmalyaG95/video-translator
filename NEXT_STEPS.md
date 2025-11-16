# Next Steps Guide

## Current Status ✅

- ✅ **Disk space cleaned**: Freed ~25GB from Docker builds
- ✅ **Docker image built**: `translate-v_python-ml-v2` (16.2GB) - created 4 minutes ago
- ⏳ **Build processes**: Multiple builds running (may be completing)
- ❌ **Services**: Not running yet

## Next Steps

### Step 1: Verify Build Completed

Check if the build finished successfully:

```bash
# Check if build processes are still running
ps aux | grep "docker.*build.*python-ml-v2" | grep -v grep

# If no output, build is complete!
# Verify the image exists
docker images | grep python-ml-v2
```

### Step 2: Start the Services

Once the build is complete, start all services:

```bash
cd /home/amalya/Desktop/translate-v

# Start all services in the background
docker-compose -f docker-compose.v2.yml up -d

# Or start with build (if needed)
docker-compose -f docker-compose.v2.yml up -d --build
```

### Step 3: Check Service Status

```bash
# Check if services are running
docker-compose -f docker-compose.v2.yml ps

# View logs
docker-compose -f docker-compose.v2.yml logs -f

# View logs for specific service
docker-compose -f docker-compose.v2.yml logs -f python-ml-v2
docker-compose -f docker-compose.v2.yml logs -f nestjs-api
docker-compose -f docker-compose.v2.yml logs -f frontend
```

### Step 4: Access the Application

Once services are running:

- **Frontend**: http://localhost:3000
- **API Gateway**: http://localhost:3001
- **Python ML Service**: gRPC on port 50052

### Step 5: Monitor Services

```bash
# Watch all logs in real-time
docker-compose -f docker-compose.v2.yml logs -f

# Check resource usage
docker stats

# Check service health
docker-compose -f docker-compose.v2.yml ps
```

## Quick Commands Reference

```bash
# Start services
docker-compose -f docker-compose.v2.yml up -d

# Stop services
docker-compose -f docker-compose.v2.yml down

# Restart services
docker-compose -f docker-compose.v2.yml restart

# Rebuild and start
docker-compose -f docker-compose.v2.yml up -d --build

# View logs
docker-compose -f docker-compose.v2.yml logs -f

# Check status
docker-compose -f docker-compose.v2.yml ps
```

## Troubleshooting

### If build is still running:
- Wait for it to complete (can take 10-30 minutes depending on dependencies)
- Monitor with: `./watch_build_realtime.sh watch`

### If services fail to start:
- Check logs: `docker-compose -f docker-compose.v2.yml logs`
- Check disk space: `df -h /`
- Check Docker: `docker system df`

### If you need to rebuild:
```bash
# Stop everything
docker-compose -f docker-compose.v2.yml down

# Clean and rebuild
docker-compose -f docker-compose.v2.yml build --no-cache python-ml-v2

# Start again
docker-compose -f docker-compose.v2.yml up -d
```


