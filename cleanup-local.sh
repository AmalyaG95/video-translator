#!/bin/bash

echo "🧹 Video Translator - Local Cleanup Script"
echo "=================================================="

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

# Clean system Python packages (optional - be careful!)
echo "⚠️  Python packages cleanup (optional)..."
echo "The following Python packages were installed for this project:"
echo "- fastapi, uvicorn, faster-whisper, transformers, torch, edge-tts"
echo "- ffmpeg-python, pydub, numpy, pandas, scipy, pydantic"
echo ""
echo "To remove them, run:"
echo "pip3 uninstall fastapi uvicorn faster-whisper transformers torch torchaudio edge-tts ffmpeg-python pydub opencv-python numpy pandas scipy pydantic pydantic-settings python-dotenv aiofiles httpx loguru pytest pytest-asyncio black flake8"
echo ""

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
echo "✅ Local cleanup completed!"
echo ""
echo "📋 What was cleaned:"
echo "  • Node.js dependencies (node_modules, package-lock.json)"
echo "  • Next.js build files (.next, out, dist)"
echo "  • Python cache files (__pycache__, *.pyc)"
echo "  • Application temp files (temp/, artifacts/, logs/)"
echo "  • npm cache"
echo "  • Electron cache"
echo ""
echo "⚠️  Note: Python packages are still installed globally."
echo "   Run the pip3 uninstall command above if you want to remove them."
echo ""
echo "🚀 To reinstall and run the app:"
echo "   ./setup.sh"
echo "   npm run dev"
