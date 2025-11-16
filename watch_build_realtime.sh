#!/bin/bash
# Real-time Docker Build Output Monitor
# This script shows live build progress from Docker BuildKit

SERVICE="python-ml-v2"
COMPOSE_FILE="docker-compose.v2.yml"

echo "🔍 Real-time Docker Build Monitor"
echo "================================="
echo ""

# Check if build is running
if ! pgrep -f "docker.*build.*${SERVICE}" > /dev/null; then
    echo "❌ No active build process found"
    echo ""
    echo "Starting a new build with real-time output..."
    cd /home/amalya/Desktop/translate-v
    DOCKER_BUILDKIT=1 docker-compose -f "$COMPOSE_FILE" build --progress=plain "$SERVICE"
    exit 0
fi

echo "✅ Build process detected!"
echo ""

# Method 1: Try to get output from Docker BuildKit directly
# This works by querying the buildx builder
get_buildx_output() {
    echo "📊 Attempting to get build output from Docker BuildKit..."
    echo ""
    
    # Check buildx processes
    BUILDX_PID=$(pgrep -f "docker-buildx.*buildx build" | head -1)
    if [ -n "$BUILDX_PID" ]; then
        echo "Found buildx process: PID $BUILDX_PID"
        echo ""
    fi
    
    # Try to get build progress using docker buildx inspect
    echo "=== Build Progress ==="
    # Unfortunately, Docker BuildKit doesn't expose a direct way to get output
    # from a running build, so we'll need to use alternative methods
}

# Method 2: Monitor Docker events for this build
monitor_docker_events() {
    echo "📡 Monitoring Docker events (build progress)..."
    echo "Press Ctrl+C to stop (build continues)"
    echo ""
    
    docker events --filter 'type=image' --filter 'type=container' --since 1m --format '{{.Time}} {{.Type}} {{.Action}} {{.ID}}' 2>/dev/null &
    EVENTS_PID=$!
    
    # Also show system resources
    while kill -0 $EVENTS_PID 2>/dev/null; do
        clear
        echo "=== Docker Build Monitor ==="
        echo ""
        echo "Build Status:"
        ps aux | grep -E "docker.*build.*${SERVICE}" | grep -v grep | head -1 | awk '{print "   PID: " $2 " | CPU: " $3 "% | MEM: " $4 "%"}'
        echo ""
        echo "Disk Space:"
        df -h / | tail -1 | awk '{print "   Used: " $3 " / " $2 " (" $5 ") | Free: " $4}'
        echo ""
        echo "Docker Build Cache:"
        docker system df 2>/dev/null | grep -A 5 "Build Cache" || echo "   Checking..."
        echo ""
        echo "Recent Docker Events (last 10):"
        docker events --filter 'type=image' --since 30s --format '{{.Time}} {{.Action}}' 2>/dev/null | tail -10 || echo "   No recent events"
        echo ""
        echo "Press Ctrl+C to exit"
        sleep 2
    done
}

# Method 3: Check build layers being created
check_build_layers() {
    echo "=== Checking Build Layers ==="
    
    # Get the image name
    IMAGE_NAME="translate-v_${SERVICE}"
    
    # Check if image exists (even partially)
    if docker images "$IMAGE_NAME" 2>/dev/null | grep -q "$IMAGE_NAME"; then
        echo "Image found, checking layers..."
        docker history "$IMAGE_NAME" 2>/dev/null | head -10
    else
        echo "Image not created yet (build in progress)"
    fi
}

# Method 4: Use docker buildx inspect (if available)
inspect_buildx() {
    echo "=== Docker Buildx Status ==="
    docker buildx ls 2>/dev/null
    echo ""
    
    # Try to get active builds
    echo "=== Active Builders ==="
    docker buildx inspect 2>/dev/null || echo "Cannot inspect builders"
}

# Main menu
case "${1:-watch}" in
    "events")
        monitor_docker_events
        ;;
    "layers")
        while true; do
            clear
            echo "=== Build Layers Monitor (refreshing every 2s) ==="
            check_build_layers
            echo ""
            inspect_buildx
            sleep 2
        done
        ;;
    "status")
        echo "=== Build Status ==="
        ps aux | grep -E "docker.*build.*${SERVICE}" | grep -v grep
        echo ""
        inspect_buildx
        echo ""
        check_build_layers
        echo ""
        echo "Disk Space:"
        df -h / | tail -1
        echo ""
        echo "Docker Resources:"
        docker system df 2>/dev/null | head -10
        ;;
    "watch"|*)
        # Continuous monitoring with auto-refresh
        while pgrep -f "docker.*build.*${SERVICE}" > /dev/null; do
            clear
            echo "=== Real-time Build Monitor (auto-refresh every 1s) ==="
            echo "Press Ctrl+C to exit"
            echo ""
            
            # Build process info
            BUILD_PROC=$(ps aux | grep -E "docker.*build.*${SERVICE}" | grep -v grep | head -1)
            if [ -n "$BUILD_PROC" ]; then
                echo "✅ Build RUNNING"
                echo "$BUILD_PROC" | awk '{print "   PID: " $2 " | CPU: " $3 "% | MEM: " $4 "% | Time: " $10}'
            else
                echo "❌ Build process not found"
                break
            fi
            echo ""
            
            # System resources
            echo "=== System Resources ==="
            df -h / | tail -1 | awk '{print "Disk: " $3 " / " $2 " (" $5 " used) | Free: " $4}'
            free -h | grep Mem | awk '{print "RAM:  " $3 " / " $2 " | Free: " $7}'
            echo ""
            
            # Docker status
            echo "=== Docker Status ==="
            docker system df 2>/dev/null | grep -E "TYPE|Build Cache" | head -2
            echo ""
            
            # Try to show image progress
            IMAGE_NAME="translate-v_${SERVICE}"
            if docker images "$IMAGE_NAME" 2>/dev/null | grep -q "$IMAGE_NAME"; then
                echo "✅ Image exists (checking layers...)"
                LAYER_COUNT=$(docker history "$IMAGE_NAME" 2>/dev/null | wc -l)
                echo "   Layers: $LAYER_COUNT"
                docker history "$IMAGE_NAME" 2>/dev/null | head -5 | tail -3
            else
                echo "⏳ Image not created yet (build in progress)"
            fi
            echo ""
            
            # Recent Docker events
            echo "=== Recent Activity (last 5 events) ==="
            docker events --filter 'type=image' --since 10s --format '{{.Time}} {{.Action}}' 2>/dev/null | tail -5 || echo "   No recent events"
            
            sleep 1
        done
        
        echo ""
        echo "Build process ended. Checking final status..."
        sleep 1
        check_build_layers
        ;;
esac


