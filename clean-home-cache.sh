#!/bin/bash
# Script to safely clean home directory caches to free space for Docker

set -e

echo "🧹 Home Directory Cache Cleanup"
echo "================================"
echo ""

# Show current space
echo "📊 Current free space:"
df -h /home/amalya | tail -1
echo ""

# Calculate potential space to free
echo "💾 Cache directories found:"
echo ""

TOTAL_POTENTIAL=0

if [ -d "/home/amalya/.cache" ]; then
    CACHE_SIZE=$(du -sh /home/amalya/.cache 2>/dev/null | cut -f1)
    CACHE_SIZE_BYTES=$(du -sb /home/amalya/.cache 2>/dev/null | cut -f1)
    echo "  📁 .cache: $CACHE_SIZE"
    TOTAL_POTENTIAL=$((TOTAL_POTENTIAL + CACHE_SIZE_BYTES))
fi

if [ -d "/home/amalya/.npm" ]; then
    NPM_SIZE=$(du -sh /home/amalya/.npm 2>/dev/null | cut -f1)
    NPM_SIZE_BYTES=$(du -sb /home/amalya/.npm 2>/dev/null | cut -f1)
    echo "  📁 .npm: $NPM_SIZE"
    TOTAL_POTENTIAL=$((TOTAL_POTENTIAL + NPM_SIZE_BYTES))
fi

if [ -d "/home/amalya/.gradle" ]; then
    GRADLE_SIZE=$(du -sh /home/amalya/.gradle 2>/dev/null | cut -f1)
    GRADLE_SIZE_BYTES=$(du -sb /home/amalya/.gradle 2>/dev/null | cut -f1)
    echo "  📁 .gradle: $GRADLE_SIZE"
    TOTAL_POTENTIAL=$((TOTAL_POTENTIAL + GRADLE_SIZE_BYTES))
fi

if [ -d "/home/amalya/.local/lib/python3" ]; then
    PYTHON_SIZE=$(du -sh /home/amalya/.local 2>/dev/null | cut -f1)
    echo "  📁 .local (Python packages): $PYTHON_SIZE"
fi

echo ""
echo "🎯 Safe to clean (will regenerate):"
echo "  1. .cache (~7.3GB) - System and app caches"
echo "  2. .npm (~5.1GB) - Node.js package cache"
echo "  3. .gradle (~6.8GB) - Gradle build cache (if not developing Android)"
echo ""
echo "⚠️  Be careful:"
echo "  - .local - Contains pip packages, might be used by projects"
echo "  - .var - Flatpak apps, some data might be important"
echo ""

# Calculate estimated space after cleanup
ESTIMATED_FREE=$((TOTAL_POTENTIAL / 1024 / 1024 / 1024))
echo "💰 Estimated space to free: ~${ESTIMATED_FREE}GB"
echo ""

read -p "Clean .cache directory? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🧹 Cleaning .cache..."
    rm -rf /home/amalya/.cache/*
    echo "✅ .cache cleaned"
fi

echo ""
read -p "Clean .npm cache? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🧹 Cleaning .npm..."
    rm -rf /home/amalya/.npm/*
    echo "✅ .npm cleaned"
    # Also clean npm cache via npm command if available
    if command -v npm &> /dev/null; then
        echo "🧹 Running npm cache clean..."
        npm cache clean --force 2>/dev/null || true
    fi
fi

echo ""
read -p "Clean .gradle cache? (WARNING: Only if not developing Android) (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🧹 Cleaning .gradle..."
    rm -rf /home/amalya/.gradle/caches/*
    echo "✅ .gradle/caches cleaned (kept other gradle files)"
fi

echo ""
echo "📊 Updated free space:"
df -h /home/amalya | tail -1
echo ""

if [ -d "/home/amalya/.local/share/pip" ]; then
    PIP_CACHE=$(du -sh /home/amalya/.local/share/pip 2>/dev/null | cut -f1)
    echo ""
    echo "💡 Optional: Clean pip cache (~$PIP_CACHE)"
    echo "   Command: pip cache purge"
    echo ""
fi

echo "✅ Cleanup complete!"
echo ""
echo "🎯 You should now have enough space for Docker deployment."


