#!/bin/bash
# Show Real-time Docker Build Output
# This script provides the best way to see live build progress

SERVICE="python-ml-v2"
COMPOSE_FILE="docker-compose.v2.yml"

echo "🔍 Real-time Docker Build Output"
echo "================================"
echo ""

# Check if build is running
if pgrep -f "docker.*build.*${SERVICE}" > /dev/null; then
    echo "⚠️  Build is already running!"
    echo ""
    echo "To see real-time output, you have two options:"
    echo ""
    echo "Option 1: Stop current build and restart with visible output"
    echo "  pkill -f 'docker.*build.*${SERVICE}'"
    echo "  cd /home/amalya/Desktop/translate-v"
    echo "  DOCKER_BUILDKIT=1 docker-compose -f $COMPOSE_FILE build --progress=plain $SERVICE"
    echo ""
    echo "Option 2: Monitor the running build (limited info)"
    echo "  ./watch_build_realtime.sh watch"
    echo ""
    echo "Current build status:"
    ps aux | grep -E "docker.*build.*${SERVICE}" | grep -v grep | head -1
    echo ""
    
    # Show what we can see
    echo "=== What we can monitor ==="
    echo "Build process:"
    ps aux | grep -E "docker.*build.*${SERVICE}" | grep -v grep | awk '{print "  PID: " $2 " | CPU: " $3 "% | MEM: " $4 "% | Runtime: " $10}'
    echo ""
    echo "Disk usage:"
    df -h / | tail -1
    echo ""
    echo "Docker build cache:"
    docker system df 2>/dev/null | grep -A 3 "Build Cache" || echo "  Checking..."
    
    exit 0
fi

# If no build is running, start one with real-time output
echo "Starting build with real-time output..."
echo ""

cd /home/amalya/Desktop/translate-v

# Use --progress=plain to see all output in real-time
# This is the key to seeing actual build logs
DOCKER_BUILDKIT=1 docker-compose -f "$COMPOSE_FILE" build --progress=plain "$SERVICE"


