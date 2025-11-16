# Docker Deployment - Space Requirements

## Current Situation
- **Project size**: ~58MB (source code only)
- **Available free space**: Check with `df -h .`
- **Minimum required**: 10-15GB for initial Docker build

## Space Breakdown

### During Build Process
1. **Base Images**: ~500MB
   - `python:3.11-slim`: ~124MB
   - `node:20-alpine`: ~200MB (x2 for frontend + nestjs)
   
2. **Python ML Service Build** (largest):
   - PyTorch: ~2-3GB
   - Transformers: ~500MB
   - Faster-Whisper: ~200MB
   - Other packages: ~500MB
   - **Build intermediate layers**: ~2-3GB (can be cleaned)
   - **Total during build**: ~5-7GB

3. **Frontend Build**:
   - Node modules: ~200-300MB
   - Build cache: ~100MB
   - **Total**: ~300-400MB

4. **NestJS Build**:
   - Node modules: ~150MB
   - Build output: ~50MB
   - **Total**: ~200MB

5. **Build Cache & Intermediate Layers**:
   - Docker layer cache: ~2-3GB (reusable)
   - Build context copies: ~500MB

### Final Images (After Build)
- **Python ML**: ~3-4GB (includes all dependencies)
- **Frontend**: ~300MB
- **NestJS**: ~200MB
- **Total running images**: ~4-5GB

### Runtime Data (Variable)
- Video uploads: depends on usage
- Processing artifacts: depends on usage
- Temporary work: depends on active processing

## Space Optimization Tips

### If Low on Space (< 10GB free):

1. **Clean Docker before build**:
   ```bash
   docker system prune -a --volumes
   # This removes unused images, containers, volumes
   # Can free 5-10GB+ if you have old images
   ```

2. **Build with no cache** (slower but smaller):
   ```bash
   docker-compose build --no-cache
   ```

3. **Build one service at a time**:
   ```bash
   docker-compose build frontend
   docker-compose build nestjs-api
   docker-compose build python-ml
   ```

4. **Remove intermediate build artifacts**:
   - After build completes, Docker keeps intermediate layers
   - Use `docker image prune` to clean up

5. **Use multi-stage builds** (already implemented):
   - Our Dockerfiles use multi-stage builds
   - Final images are optimized

### Recommended Free Space
- **Minimum**: 10GB (tight, might fail on large builds)
- **Recommended**: 15-20GB (comfortable)
- **Ideal**: 25GB+ (for long-term operation)

## Check Your Current Space

```bash
# Check free space
df -h .

# Check Docker disk usage
docker system df

# Check project size
du -sh /home/amalya/Desktop/translate-v
```

## If You're Short on Space

### Option 1: Clean Docker System
```bash
# Warning: Removes unused images, containers, networks, volumes
docker system prune -a --volumes

# Check space freed
df -h .
```

### Option 2: Build on External Drive
```bash
# Move project to external drive with more space
# Or change Docker data directory (advanced)
```

### Option 3: Remove Unused Software
- Clean package manager cache
- Remove unused applications
- Move large files to external storage

### Option 4: Use Docker BuildKit with GC
```bash
DOCKER_BUILDKIT=1 docker-compose build
```

## Post-Build Cleanup

After successful build, you can reduce space:

```bash
# Remove build cache (keeps final images)
docker builder prune

# Check what's taking space
docker images
docker system df
```

## Monitoring During Build

```bash
# Watch disk space during build
watch -n 5 df -h .

# Or in another terminal
docker system df
```

## Expected Build Results

After successful build:
- **3 Docker images**: ~4-5GB total
- **Build cache**: ~2-3GB (optional to keep)
- **Total**: ~6-8GB (can reduce to 4-5GB after cleanup)

## Runtime Requirements

- **Base images**: ~4-5GB
- **Volume data**: Variable (uploads, artifacts)
- **Recommended free**: 5-10GB for runtime operations


