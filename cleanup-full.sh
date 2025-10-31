#!/bin/bash

echo "🧹 Video Translator - FULL Cleanup Script"
echo "=================================================="
echo "⚠️  WARNING: This will remove ALL Python packages installed for this project!"
echo ""

read -p "Are you sure you want to continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Cleanup cancelled."
    exit 1
fi

# Stop any running processes
echo "🛑 Stopping running processes..."
pkill -f "uvicorn main:sio_app" 2>/dev/null || true
pkill -f "next dev" 2>/dev/null || true
pkill -f "electron" 2>/dev/null || true
pkill -f "concurrently" 2>/dev/null || true

# Clean Node.js dependencies and cache
echo "📦 Cleaning Node.js dependencies..."
rm -rf node_modules
rm -f package-lock.json
rm -rf .next
rm -rf out
rm -rf dist

# Clean npm cache
echo "🗑️  Cleaning npm cache..."
npm cache clean --force 2>/dev/null || true

# Clean Python cache and temporary files
echo "🐍 Cleaning Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "*.pyo" -delete 2>/dev/null || true

# Clean application temporary files from .data folder
echo "📁 Cleaning application files..."
rm -rf .data/temp_work/*
rm -rf .data/artifacts/*
rm -rf .data/uploads/*
rm -rf .data/logs/*
rm -rf backend/static/*

# Remove Python packages
echo "🐍 Removing Python packages..."
pip3 uninstall -y fastapi uvicorn faster-whisper transformers torch torchaudio edge-tts ffmpeg-python pydub opencv-python numpy pandas scipy pydantic pydantic-settings python-dotenv aiofiles httpx loguru pytest pytest-asyncio black flake8 2>/dev/null || true

# Clean Electron cache
echo "⚡ Cleaning Electron cache..."
rm -rf ~/.cache/electron 2>/dev/null || true

# Clean system temp files
echo "🗂️  Cleaning system temp files..."
rm -rf /tmp/electron-* 2>/dev/null || true
rm -rf /tmp/video-translator-* 2>/dev/null || true

# Show disk space freed
echo "💾 Disk space check..."
df -h /home/amalya/Desktop/video-translator

echo ""
echo "✅ FULL cleanup completed!"
echo ""
echo "📋 What was cleaned:"
echo "  • Node.js dependencies (node_modules, package-lock.json)"
echo "  • Next.js build files (.next, out, dist)"
echo "  • Python cache files (__pycache__, *.pyc)"
echo "  • Application temp files (temp/, artifacts/, logs/)"
echo "  • npm cache"
echo "  • Electron cache"
echo "  • ALL Python packages installed for this project"
echo ""
echo "🚀 To reinstall and run the app:"
echo "   ./setup.sh"
echo "   npm run dev"
