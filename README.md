# Video Translation System

A video translation system that dubs videos while preserving exact duration and natural lip-sync.

## Setup (First Time Only)

Before running the application, install all dependencies:

```bash
./setup.sh
```

This will:

- Install Python packages for ML service
- Install Node.js packages for API and Frontend
- Create necessary directories
- Check for required tools (Python, Node.js, FFmpeg)

## Quick Start

### Option 1: Local Development

```bash
./start-app.sh
./start-electron.sh
```

This starts everything:

- Python ML service (port 50051)
- NestJS API (port 3001)
- Next.js Frontend (port 3000)
- Electron desktop app

### Option 2: Docker

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Option 4: Manual Setup

```bash
# 1. Python ML Service
cd backend-python-ml
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
python src/main.py

# 2. NestJS API (new terminal)
cd backend-nestjs
npm install
npm run build
npm run start:dev

# 3. Frontend (new terminal)
cd frontend
npm install
npm run dev
```

## Access URLs

- **Frontend**: http://localhost:3000
- **NestJS API**: http://localhost:3001
- **Python ML**: localhost:50051 (gRPC)
- **Electron**: Launches with command ./start-electron.sh

## Requirements

- Python 3.11+
- Node.js 18+
- FFmpeg
- 8GB+ RAM
- 20GB+ free space

## Features

- Real-time progress tracking
- Quality metrics (lip-sync, voice, translation)
- Early preview during processing
- Complete session history
- Desktop app support

---
