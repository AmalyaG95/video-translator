# How to Check Docker Build Logs

## Quick Commands

### 1. View Build Output in Real-Time
```bash
# Build with verbose output (shows all steps)
docker-compose build python-ml-v2

# Or with no-cache to see full build process
docker-compose build --no-cache python-ml-v2
```

### 2. Save Build Logs to File
```bash
# Save build output to a file
docker-compose build python-ml-v2 2>&1 | tee build.log

# Then view the log
cat build.log
# or
less build.log
```

### 3. Check Docker Build History
```bash
# See build history for the service
docker history translate-v-python-ml-v2:latest

# Or if using docker-compose project name
docker history $(docker-compose ps -q python-ml-v2 2>/dev/null | head -1)
```

### 4. Inspect Failed Build Layers
```bash
# List all images (including failed builds)
docker images -a

# Inspect a specific image
docker inspect translate-v-python-ml-v2:latest
```

### 5. Check Build Cache
```bash
# See what's in the build cache
docker buildx du

# Or check system disk usage
docker system df -v
```

## For Your Specific Service (python-ml-v2)

### Build with Detailed Output
```bash
cd /home/amalya/Desktop/translate-v
docker-compose -f docker-compose.v2.yml build --progress=plain python-ml-v2
```

The `--progress=plain` flag shows all build output without fancy formatting.

### Build and Save Logs
```bash
docker-compose -f docker-compose.v2.yml build python-ml-v2 2>&1 | tee python-ml-v2-build.log
```

### Check What Failed
```bash
# View the last 100 lines of build output
tail -100 python-ml-v2-build.log

# Search for errors
grep -i error python-ml-v2-build.log
grep -i fail python-ml-v2-build.log
```

## Using Docker Build Directly (More Control)

If docker-compose is having issues, you can build directly:

```bash
cd /home/amalya/Desktop/translate-v/backend-python-ml-v2
docker build -t python-ml-v2:latest . 2>&1 | tee ../build-direct.log
```

## View Container Logs (After Build Succeeds)

Once the container is running:
```bash
# View logs from running container
docker-compose -f docker-compose.v2.yml logs python-ml-v2

# Follow logs in real-time
docker-compose -f docker-compose.v2.yml logs -f python-ml-v2

# View last 100 lines
docker-compose -f docker-compose.v2.yml logs --tail=100 python-ml-v2
```

## Debugging Build Issues

### Check Disk Space (Your Current Issue)
```bash
df -h /
docker system df
```

### Clean Up Before Rebuilding
```bash
# Remove old build cache
docker builder prune -a

# Remove unused images
docker image prune -a

# Full cleanup (be careful!)
docker system prune -a
```

### Build with Verbose Output
```bash
DOCKER_BUILDKIT=0 docker-compose build python-ml-v2
```

The `DOCKER_BUILDKIT=0` disables BuildKit and shows traditional build output.

## Common Issues and Solutions

### "no space left on device"
- Run the cleanup script: `./cleanup_disk.sh`
- Clean Docker: `docker system prune -a`

### Build hangs or is slow
- Check disk I/O: `iostat -x 1`
- Check memory: `free -h`
- Build with limited parallelism: `docker-compose build --parallel 1 python-ml-v2`

### Want to see each step
```bash
docker-compose build --progress=plain --no-cache python-ml-v2
```


