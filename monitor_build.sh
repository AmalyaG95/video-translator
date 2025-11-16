#!/bin/bash
# Real-time Docker Build Progress Monitor

echo "=== Docker Build Progress Monitor ==="
echo ""

# Check if build is running
if ! pgrep -f "docker.*build.*python-ml-v2" > /dev/null; then
    echo "⚠️  No active build process found for python-ml-v2"
    echo "Starting a new build with logging..."
    cd /home/amalya/Desktop/translate-v
    docker-compose -f docker-compose.v2.yml build python-ml-v2 2>&1 | tee /tmp/docker-build.log
    exit 0
fi

echo "✅ Build process detected!"
echo ""

# Function to show current progress
show_progress() {
    if [ -f /tmp/docker-build.log ]; then
        echo "=== Latest Build Output (last 20 lines) ==="
        tail -20 /tmp/docker-build.log
        echo ""
        echo "=== Build Statistics ==="
        echo "Total lines in log: $(wc -l < /tmp/docker-build.log)"
        echo "Errors found: $(grep -i "error\|fail" /tmp/docker-build.log | wc -l)"
        echo ""
    fi
}

# Function to monitor in real-time
monitor_realtime() {
    echo "📊 Monitoring build progress in real-time..."
    echo "Press Ctrl+C to stop monitoring (build will continue)"
    echo ""
    
    if [ -f /tmp/docker-build.log ]; then
        tail -f /tmp/docker-build.log
    else
        echo "Waiting for log file to be created..."
        while [ ! -f /tmp/docker-build.log ]; do
            sleep 1
        done
        tail -f /tmp/docker-build.log
    fi
}

# Function to show build status
show_status() {
    echo "=== Build Process Status ==="
    ps aux | grep -E "docker.*build.*python-ml-v2" | grep -v grep | head -3
    echo ""
    
    # Check Docker buildx status
    if command -v docker &> /dev/null; then
        echo "=== Docker Buildx Status ==="
        docker buildx ls 2>/dev/null || echo "Buildx not available"
        echo ""
    fi
}

# Main menu
case "${1:-monitor}" in
    "status")
        show_status
        show_progress
        ;;
    "tail")
        monitor_realtime
        ;;
    "errors")
        if [ -f /tmp/docker-build.log ]; then
            echo "=== Errors/Warnings in Build ==="
            grep -i "error\|fail\|warning" /tmp/docker-build.log | tail -20
        else
            echo "Log file not found"
        fi
        ;;
    "full")
        if [ -f /tmp/docker-build.log ]; then
            cat /tmp/docker-build.log
        else
            echo "Log file not found"
        fi
        ;;
    "monitor"|*)
        show_status
        show_progress
        echo ""
        echo "Usage:"
        echo "  ./monitor_build.sh          - Show current status and progress"
        echo "  ./monitor_build.sh tail     - Follow build output in real-time"
        echo "  ./monitor_build.sh status   - Show build process status"
        echo "  ./monitor_build.sh errors   - Show only errors/warnings"
        echo "  ./monitor_build.sh full     - Show full build log"
        ;;
esac


