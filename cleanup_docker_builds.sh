#!/bin/bash
# Clean up Docker build data and cache

echo "🧹 Docker Build Data Cleanup"
echo "============================"
echo ""

# Show current usage
echo "📊 Current Docker disk usage:"
docker system df
echo ""

# Ask for confirmation (optional - can be automated)
if [ "${1:-auto}" != "auto" ]; then
    read -p "Continue with cleanup? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cleanup cancelled."
        exit 0
    fi
fi

echo "Starting cleanup..."
echo ""

# 1. Remove unused images (dangling and unused)
echo "1. Removing unused images..."
docker image prune -a -f
echo "   ✓ Done"
echo ""

# 2. Remove build cache
echo "2. Removing build cache..."
docker builder prune -a -f
echo "   ✓ Done"
echo ""

# 3. Remove stopped containers
echo "3. Removing stopped containers..."
docker container prune -f
echo "   ✓ Done"
echo ""

# 4. Remove unused volumes
echo "4. Removing unused volumes..."
docker volume prune -f
echo "   ✓ Done"
echo ""

# 5. Remove unused networks
echo "5. Removing unused networks..."
docker network prune -f
echo "   ✓ Done"
echo ""

# 6. Full system cleanup (everything unused)
echo "6. Performing full system cleanup..."
docker system prune -a -f --volumes
echo "   ✓ Done"
echo ""

# Show final usage
echo "📊 Final Docker disk usage:"
docker system df
echo ""

# Calculate space freed
echo "✅ Cleanup complete!"
echo ""
echo "💡 To see detailed breakdown:"
echo "   docker system df -v"


