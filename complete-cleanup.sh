#!/bin/bash

echo "🧹 Complete Cleanup Script"
echo "=========================="

# Kill all running processes
echo "1. Killing all running processes..."
pkill -f "nest\|next\|node" 2>/dev/null || true
sleep 2

# Clear all video files
echo "2. Clearing all video files..."
sudo find . -name "*.mp4" -type f -delete 2>/dev/null || true
sudo find . -name "*.avi" -type f -delete 2>/dev/null || true
sudo find . -name "*.mov" -type f -delete 2>/dev/null || true
sudo find . -name "*.mkv" -type f -delete 2>/dev/null || true

# Clear all session data from .data folder
echo "3. Clearing all session data..."
rm -rf .data/artifacts/* 2>/dev/null || true
rm -rf .data/uploads/* 2>/dev/null || true
rm -rf .data/temp_work/* 2>/dev/null || true
rm -rf backend-python-ml/outputs/* 2>/dev/null || true
rm -rf backend-old/uploads/* 2>/dev/null || true
rm -rf backend-old/outputs/* 2>/dev/null || true
rm -rf backend-old/artifacts/* 2>/dev/null || true

# Clear logs
echo "4. Clearing logs..."
rm -rf .data/logs/* 2>/dev/null || true
rm -f backend-python-ml/artifacts/logs.jsonl 2>/dev/null || true

# Clear node_modules and reinstall
echo "5. Clearing node_modules..."
rm -rf node_modules 2>/dev/null || true
rm -rf frontend/node_modules 2>/dev/null || true
rm -rf backend-nestjs/node_modules 2>/dev/null || true

# Clear package-lock files
rm -f package-lock.json 2>/dev/null || true
rm -f frontend/package-lock.json 2>/dev/null || true
rm -f backend-nestjs/package-lock.json 2>/dev/null || true

# Clear build artifacts
echo "6. Clearing build artifacts..."
rm -rf frontend/.next 2>/dev/null || true
rm -rf backend-nestjs/dist 2>/dev/null || true

# Clear Python cache
echo "7. Clearing Python cache..."
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -type f -delete 2>/dev/null || true

# Clear temporary files
echo "8. Clearing temporary files..."
rm -rf /tmp/translate-v* 2>/dev/null || true
rm -rf /tmp/video-translator-* 2>/dev/null || true

echo "✅ Complete cleanup finished!"
echo ""
echo "Next steps:"
echo "1. Run: npm install"
echo "2. Run: cd frontend && npm install"
echo "3. Run: cd backend-nestjs && npm install"
echo "4. Run: make run"
echo ""
echo "The application is now completely clean and ready for real video uploads only!"


