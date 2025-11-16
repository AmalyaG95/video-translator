#!/bin/bash
# Real-time Docker Build Progress Tracker

set -e

BUILD_LOG="/tmp/docker-build.log"
SERVICE="python-ml-v2"

echo "🔍 Docker Build Progress Tracker"
echo "================================"
echo ""

# Method 1: Check if build process is running
check_build_running() {
    if pgrep -f "docker.*build.*${SERVICE}" > /dev/null; then
        echo "✅ Build process is RUNNING"
        ps aux | grep -E "docker.*build.*${SERVICE}" | grep -v grep | head -1 | awk '{print "   PID: " $2 " | CPU: " $3 "% | MEM: " $4 "%"}'
        return 0
    else
        echo "❌ No active build process found"
        return 1
    fi
}

# Method 2: Monitor log file in real-time
monitor_log() {
    if [ -f "$BUILD_LOG" ]; then
        echo "📄 Monitoring log file: $BUILD_LOG"
        echo "   File size: $(du -h "$BUILD_LOG" | cut -f1)"
        echo "   Lines: $(wc -l < "$BUILD_LOG")"
        echo ""
        echo "=== Latest Output (last 15 lines) ==="
        tail -15 "$BUILD_LOG"
        echo ""
        echo "Press Ctrl+C to stop (build continues in background)"
        echo "Following log in real-time..."
        echo ""
        tail -f "$BUILD_LOG"
    else
        echo "⏳ Log file not created yet. Waiting..."
        echo "   (This is normal at the start of a build)"
        echo ""
        echo "Checking build process directly..."
        
        # Try to get output from docker buildx
        if command -v docker &> /dev/null; then
            echo "=== Checking Docker Buildx Status ==="
            docker buildx ls 2>/dev/null || true
        fi
    fi
}

# Method 3: Show build statistics
show_stats() {
    if [ -f "$BUILD_LOG" ]; then
        echo "=== Build Statistics ==="
        echo "Total log lines: $(wc -l < "$BUILD_LOG")"
        echo "Errors: $(grep -ci "error" "$BUILD_LOG" 2>/dev/null || echo "0")"
        echo "Warnings: $(grep -ci "warning" "$BUILD_LOG" 2>/dev/null || echo "0")"
        echo "Success steps: $(grep -ci "step.*done\|success" "$BUILD_LOG" 2>/dev/null || echo "0")"
        echo ""
        
        # Show current step
        CURRENT_STEP=$(grep -i "step\|layer\|stage" "$BUILD_LOG" | tail -1)
        if [ -n "$CURRENT_STEP" ]; then
            echo "Current step: $CURRENT_STEP"
        fi
    fi
}

# Method 4: Show errors only
show_errors() {
    if [ -f "$BUILD_LOG" ]; then
        echo "=== Errors and Warnings ==="
        grep -i "error\|fail\|warning" "$BUILD_LOG" | tail -20 || echo "No errors found so far"
    else
        echo "Log file not available yet"
    fi
}

# Method 5: Monitor system resources
monitor_resources() {
    echo "=== System Resources ==="
    echo "Disk space:"
    df -h / | tail -1 | awk '{print "   Used: " $3 " / " $2 " (" $5 " used) | Free: " $4}'
    echo ""
    echo "Memory:"
    free -h | grep Mem | awk '{print "   Used: " $3 " / " $2 " | Free: " $7}'
    echo ""
    echo "Docker disk usage:"
    docker system df 2>/dev/null | head -5 || echo "   Docker not responding"
}

# Main execution
case "${1:-monitor}" in
    "log"|"tail")
        check_build_running
        echo ""
        monitor_log
        ;;
    "status")
        check_build_running
        echo ""
        show_stats
        echo ""
        monitor_resources
        ;;
    "errors")
        show_errors
        ;;
    "stats")
        show_stats
        ;;
    "resources")
        monitor_resources
        ;;
    "watch")
        # Continuous monitoring
        while true; do
            clear
            echo "=== Docker Build Monitor (Auto-refresh every 2s) ==="
            echo "Press Ctrl+C to exit"
            echo ""
            check_build_running
            echo ""
            show_stats
            echo ""
            if [ -f "$BUILD_LOG" ]; then
                echo "=== Latest Output ==="
                tail -10 "$BUILD_LOG"
            fi
            echo ""
            monitor_resources
            sleep 2
        done
        ;;
    "monitor"|*)
        check_build_running
        echo ""
        show_stats
        echo ""
        if [ -f "$BUILD_LOG" ]; then
            echo "=== Recent Output ==="
            tail -20 "$BUILD_LOG"
        else
            echo "💡 Tip: To see real-time output, run:"
            echo "   ./track_build_progress.sh log"
        fi
        echo ""
        echo "Available commands:"
        echo "  ./track_build_progress.sh          - Quick status check"
        echo "  ./track_build_progress.sh log      - Follow log in real-time"
        echo "  ./track_build_progress.sh watch    - Auto-refresh monitor"
        echo "  ./track_build_progress.sh status   - Detailed status"
        echo "  ./track_build_progress.sh errors   - Show errors only"
        echo "  ./track_build_progress.sh stats    - Build statistics"
        echo "  ./track_build_progress.sh resources - System resources"
        ;;
esac


