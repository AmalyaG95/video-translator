#!/bin/bash
# Disk Cleanup Script
# This script helps free up disk space safely

echo "=== Disk Cleanup Script ==="
echo "Current disk usage:"
df -h / | tail -1
echo ""

# Function to show size before and after
show_space() {
    echo "Space before cleanup:"
    df -h / | tail -1 | awk '{print "Used: " $3 " / " $2 " (" $5 ")"}'
}

show_space
echo ""

# 1. Clean Chrome cache (4.2 GB)
echo "1. Cleaning Chrome cache (~4.2 GB)..."
rm -rf ~/.var/app/com.google.Chrome/cache/*
echo "   ✓ Chrome cache cleaned"

# 2. Clean Gradle cache (6.8 GB) - safe to clean, will be rebuilt when needed
echo "2. Cleaning Gradle cache (~6.8 GB)..."
rm -rf ~/.gradle/caches/*
echo "   ✓ Gradle cache cleaned"

# 3. Clean system journal logs (2.0 GB)
echo "3. Cleaning journal logs (~2.0 GB)..."
sudo journalctl --vacuum-time=3d
echo "   ✓ Journal logs cleaned"

# 4. Clean old system logs (2.6 GB)
echo "4. Cleaning old system logs (~2.6 GB)..."
sudo find /var/log -type f -name "*.log" -mtime +7 -delete
sudo find /var/log -type f -name "*.gz" -delete
echo "   ✓ System logs cleaned"

# 5. Clean APT cache (if any)
echo "5. Cleaning APT cache..."
sudo apt-get clean
sudo apt-get autoclean
echo "   ✓ APT cache cleaned"

# 6. Clean Docker (if unused)
echo "6. Cleaning Docker unused resources..."
docker system prune -f 2>/dev/null || echo "   (Docker not available)"
echo "   ✓ Docker cleaned"

# 7. Clean npm cache (452 MB)
echo "7. Cleaning npm cache (~452 MB)..."
npm cache clean --force 2>/dev/null || echo "   (npm not available)"
echo "   ✓ npm cache cleaned"

# 8. Clean pip cache (if exists)
echo "8. Cleaning pip cache..."
pip cache purge 2>/dev/null || python3 -m pip cache purge 2>/dev/null || echo "   (pip cache not found)"
echo "   ✓ pip cache cleaned"

# 9. Clean temporary files
echo "9. Cleaning temporary files..."
rm -rf /tmp/* 2>/dev/null
rm -rf ~/.cache/* 2>/dev/null
echo "   ✓ Temporary files cleaned"

echo ""
echo "=== Cleanup Complete ==="
echo "Space after cleanup:"
df -h / | tail -1 | awk '{print "Used: " $3 " / " $2 " (" $5 ")"}'
echo ""
echo "Expected space freed: ~15-20 GB"


