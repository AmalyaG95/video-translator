#!/bin/bash

# Cleanup script to free space before Docker build
# This removes all unnecessary files but keeps source code

echo "🧹 Starting cleanup to free space before Docker..."
echo ""

# Function to show size of a directory
show_size() {
    if [ -d "$1" ]; then
        size=$(du -sh "$1" 2>/dev/null | cut -f1)
        echo "  📦 $1: $size"
    fi
}

echo "📊 Current space usage:"
show_size "backend/venv"
show_size "backend/temp_work"
show_size "backend/outputs"
show_size "backend/artifacts"
show_size "frontend/node_modules"
show_size "frontend/.next"
show_size "node_modules"
show_size "backend-nestjs/node_modules"
show_size "backend-nestjs/dist"
show_size "backend-python-ml/venv"
echo ""

# Calculate total before
total_before=$(du -sh . 2>/dev/null | cut -f1)
echo "💾 Total project size BEFORE cleanup: $total_before"
echo ""

echo "🗑️  Removing unnecessary files..."
echo ""

# Remove Python virtual environments (will be recreated in Docker)
echo "  → Removing Python venv (backend/venv)..."
rm -rf backend/venv

echo "  → Removing Python venv (backend-python-ml/venv)..."
rm -rf backend-python-ml/venv

# Remove Python cache files
echo "  → Removing Python __pycache__..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# Remove temporary work files
echo "  → Removing temporary work files..."
rm -rf backend/temp_work/*
rm -rf backend/outputs/*
rm -rf temp/*
rm -rf temp_work/*

# Remove artifacts (will be regenerated)
echo "  → Removing artifacts..."
rm -rf backend/artifacts/*
rm -rf artifacts/*

# Remove old uploads
echo "  → Removing old uploads..."
rm -rf backend/uploads/*
rm -rf uploads/*

# Remove Node.js dependencies (will be installed in Docker)
echo "  → Removing node_modules (root)..."
rm -rf node_modules

echo "  → Removing node_modules (frontend)..."
rm -rf frontend/node_modules
rm -rf frontend/.next

echo "  → Removing node_modules (backend-nestjs)..."
rm -rf backend-nestjs/node_modules
rm -rf backend-nestjs/dist

# Remove test outputs
echo "  → Removing test outputs..."
rm -rf tts_test_outputs
rm -rf test_*.wav

# Remove logs
echo "  → Removing logs..."
rm -rf logs/*
rm -f backend/*.log

# Remove any .pyc files
echo "  → Removing .pyc files..."
find . -type f -name "*.pyc" -delete 2>/dev/null

echo ""
echo "✅ Cleanup complete!"
echo ""

# Calculate total after
total_after=$(du -sh . 2>/dev/null | cut -f1)
echo "💾 Total project size AFTER cleanup: $total_after"
echo ""

echo "📊 Remaining size by directory:"
du -sh */ 2>/dev/null | sort -hr | head -10
echo ""

echo "✨ Your project is now lighter and ready for Docker!"
echo ""
echo "🚀 Next steps:"
echo "   1. Run: make docker-build"
echo "   2. Run: make docker-up"
echo ""
