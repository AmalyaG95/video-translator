#!/bin/bash
# Script to free up space before Docker deployment

set -e

echo "🔍 Freeing Space for Docker Deployment"
echo "======================================"
echo ""

# Check current space
echo "📊 Current disk space:"
df -h . | tail -1
echo ""

# Check Docker usage
echo "🐳 Docker disk usage:"
docker system df 2>/dev/null || echo "Docker not accessible"
echo ""

# Options for freeing space
echo "💡 Space-Freeing Options:"
echo ""
echo "1. Clean Docker system (recommended first step)"
echo "   Command: docker system prune -a --volumes"
echo "   Can free: 1-10GB+ (depends on what's cached)"
echo ""

echo "2. Clean Docker build cache"
echo "   Command: docker builder prune -a"
echo "   Can free: 1-5GB"
echo ""

echo "3. Remove unused Docker images"
echo "   Command: docker image prune -a"
echo "   Can free: varies"
echo ""

echo "4. Check large files/directories"
echo "   Command: du -h --max-depth=1 ~ | sort -hr | head -20"
echo ""

read -p "Do you want to clean Docker system now? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🧹 Cleaning Docker system..."
    docker system prune -a --volumes --force
    echo ""
    echo "✅ Docker cleanup complete!"
    echo ""
    echo "📊 Updated disk space:"
    df -h . | tail -1
    echo ""
    echo "Check if you now have enough space (need ~10GB minimum)"
fi

echo ""
echo "If still insufficient, consider:"
echo "- Moving project to external drive"
echo "- Cleaning package manager caches"
echo "- Removing unused applications"
echo "- Compressing old files"


