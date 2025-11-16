# Electron Desktop App Deployment Guide

This guide explains how to deploy and run the Electron desktop app after Docker services are running.

## Prerequisites

1. **Docker services must be running**:
   ```bash
   ./deploy.sh up
   ```

2. **Services must be accessible**:
   - Frontend: http://localhost:3000
   - NestJS API: http://localhost:3001
   - Python ML: localhost:50051 (gRPC)

## Quick Start

### Option 1: Automated Deployment Script

```bash
# After Docker services are running
./deploy-electron.sh
```

This script will:
- ✅ Check if all Docker services are running
- ✅ Verify service connectivity
- ✅ Install Electron dependencies if needed
- ✅ Launch the Electron desktop app

### Option 2: Manual Launch

```bash
# 1. Ensure services are running
./deploy.sh status

# 2. Install Electron if not already installed
npm install

# 3. Launch Electron
NODE_ENV=development npx electron .
```

## How It Works

The Electron app connects to Docker services running on localhost:

```
Electron App → http://localhost:3000 (Frontend)
            → http://localhost:3001 (NestJS API)
            → localhost:50051 (Python ML gRPC)
```

The app loads the Next.js frontend from the Docker container and communicates with backend services.

## Configuration

### Development Mode (Docker)

The Electron app is configured to:
- Load frontend from: `http://localhost:3000`
- Connect to API at: `http://localhost:3001`
- Use gRPC for ML service: `localhost:50051`

### Production Mode (Standalone)

For standalone builds (not using Docker):
```bash
# Build Next.js for production
cd frontend && npm run build

# Package Electron app
npm run build:electron
```

## Troubleshooting

### Services Not Running

**Error**: "Frontend is not running" or "NestJS API is not running"

**Solution**:
```bash
# Start Docker services
./deploy.sh up

# Check status
./deploy.sh status

# View logs
./deploy.sh logs
```

### Electron Won't Launch

**Error**: "Cannot find module 'electron'"

**Solution**:
```bash
npm install
```

### Can't Connect to Services

**Issue**: Electron can't reach localhost services

**Possible causes**:
1. Docker services not running - check with `./deploy.sh status`
2. Firewall blocking localhost
3. Services running on different ports

**Solution**:
```bash
# Verify services are accessible
curl http://localhost:3000
curl http://localhost:3001/health

# Check Docker containers
docker ps
```

### CORS Errors

If you see CORS errors, ensure:
- Docker services are configured to allow Electron origin
- Frontend API URL is set to `http://localhost:3001`

## Building Standalone App

To create a standalone executable (doesn't require Docker):

```bash
# 1. Build frontend
cd frontend
npm run build

# 2. Build Electron package
cd ..
npm run build:electron
```

Output will be in `dist/` directory:
- **Linux**: `dist/Video Translator-*.AppImage`
- **Windows**: `dist/Video Translator Setup *.exe`
- **macOS**: `dist/Video Translator-*.dmg`

## Development Workflow

### With Docker Services

1. Start Docker services:
   ```bash
   ./deploy.sh up
   ```

2. Launch Electron:
   ```bash
   ./deploy-electron.sh
   ```

3. Make changes - Electron will reload when you refresh (F5)

### Without Docker (Local Development)

1. Start all services locally:
   ```bash
   ./start-app.sh
   ```

2. Launch Electron:
   ```bash
   npm run dev:electron
   ```

## Features

- ✅ Native desktop window
- ✅ System tray support (if configured)
- ✅ Auto-updates (if configured)
- ✅ File system access
- ✅ Native dialogs

## Next Steps

After launching:
1. The Electron window should open
2. You should see the Video Translator UI
3. You can upload videos and start translations
4. All processing happens via Docker services

## Support

For issues:
1. Check Docker services: `./deploy.sh logs`
2. Check Electron console (DevTools)
3. Review `ELECTRON_DEPLOYMENT.md` for common issues


