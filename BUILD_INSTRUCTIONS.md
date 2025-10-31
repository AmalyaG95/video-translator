# 🔧 Docker Build Instructions (Fixed)

## Issue Fixed
✅ Added missing dependencies (pkg-config, FFmpeg dev libraries)
✅ Added non-interactive mode to avoid debconf errors
✅ Cleaned up Dockerfile

## To Build Docker Images

### Option 1: Build Everything (Recommended)
Open your terminal and run:

```bash
cd /home/amalya/Desktop/translate-v
make docker-build
```

This will build all 3 services:
- Python ML (takes ~10 min - longest due to PyAV compilation)
- NestJS API (takes ~3-5 min)
- Frontend (takes ~3-5 min)

**Total time: ~15-20 minutes**

### Option 2: Build Step by Step
If the above doesn't work or you want more control:

```bash
cd /home/amalya/Desktop/translate-v

# Build Python ML service (slowest)
./docker-with-permissions.sh build python-ml

# Build NestJS service
./docker-with-permissions.sh build nestjs

# Build Frontend service
./docker-with-permissions.sh build frontend
```

### Option 3: Use the rebuild script
```bash
cd /home/amalya/Desktop/translate-v
./rebuild-docker.sh
```

## After Build Completes

Start the services:
```bash
make docker-up
```

Then run Electron:
```bash
npm run dev:electron
```

## If Build Fails

1. **Clean everything first:**
   ```bash
   make clean
   ```

2. **Try building again:**
   ```bash
   make docker-build
   ```

3. **Check logs if it fails:**
   ```bash
   make docker-logs
   ```

## Troubleshooting

### "debconf" errors
✅ FIXED - Added DEBIAN_FRONTEND=noninteractive

### "pkg-config is required"
✅ FIXED - Added pkg-config and FFmpeg dev libraries

### "Permission denied"
✅ FIXED - Using docker-with-permissions.sh wrapper

### Build takes too long
⏱️ NORMAL - First build takes 15-20 minutes
   - Python ML: ~10 min (compiling PyAV)
   - NestJS: ~3-5 min
   - Frontend: ~3-5 min
   
   Subsequent builds are much faster (use cache)!

## What's Being Built

1. **Python ML Container:**
   - Python 3.11
   - FFmpeg + development libraries
   - PyTorch, transformers, faster-whisper
   - gRPC server
   - PyAV (compiled from source - takes longest)

2. **NestJS Container:**
   - Node.js 20
   - NestJS framework
   - TypeScript compiled
   - gRPC client

3. **Frontend Container:**
   - Node.js 20
   - Next.js 15
   - React dependencies

## Ready to Start!

Just run in your terminal:
```bash
cd /home/amalya/Desktop/translate-v
make docker-build
```

And wait ~15-20 minutes. ☕

Then:
```bash
make docker-up
npm run dev:electron
```

🎉 Desktop app will open!
