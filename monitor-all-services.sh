#!/bin/bash

# Comprehensive Service Monitoring Script
# Monitors all services and displays logs in real-time

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Service configurations
# Try both .data/logs (from start-app.sh) and logs/ directories
SERVICES=(
    "nestjs:NestJS API:3001:.data/logs/nestjs-api.log:logs/nestjs-api.log"
    "python-ml:Python ML:50051:.data/logs/python-ml.log:logs/python-ml.log"
    "python-http:Python HTTP:50052:.data/logs/python-http.log:logs/python-http.log"
    "frontend:Frontend:3000:.data/logs/frontend.log:logs/frontend.log"
)

# Function to check if service is running
check_service() {
    local service_name=$1
    local port=$2
    
    # Try multiple methods to check if port is listening
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} $service_name is running on port $port"
        return 0
    elif ss -tlnp | grep -q ":$port "; then
        echo -e "${GREEN}✓${NC} $service_name is running on port $port"
        return 0
    elif netstat -tlnp 2>/dev/null | grep -q ":$port "; then
        echo -e "${GREEN}✓${NC} $service_name is running on port $port"
        return 0
    else
        echo -e "${RED}✗${NC} $service_name is not running on port $port"
        return 1
    fi
}

# Function to display service status
show_status() {
    echo -e "\n${BLUE}=== SERVICE STATUS ===${NC}"
    local all_running=true
    
    for service_config in "${SERVICES[@]}"; do
        IFS=':' read -r name display_name port primary_log fallback_log <<< "$service_config"
        if ! check_service "$display_name" "$port"; then
            all_running=false
        fi
    done
    
    if [ "$all_running" = true ]; then
        echo -e "\n${GREEN}All services are running!${NC}"
    else
        echo -e "\n${YELLOW}Some services are not running. Check the logs below.${NC}"
    fi
}

# Function to find log file (check multiple locations)
find_log_file() {
    local primary_log=$1
    local fallback_log=$2
    
    if [ -f "$primary_log" ]; then
        echo "$primary_log"
        return 0
    elif [ -f "$fallback_log" ]; then
        echo "$fallback_log"
        return 0
    else
        return 1
    fi
}

# Function to display logs for a specific service
show_logs() {
    local service_name=$1
    local primary_log=$2
    local fallback_log=$3
    local color=$4
    
    local log_file
    if log_file=$(find_log_file "$primary_log" "$fallback_log"); then
        echo -e "\n${color}=== $service_name LOGS (last 20 lines from $log_file) ===${NC}"
        tail -20 "$log_file" | sed "s/^/${color}[$service_name]${NC} /"
    else
        echo -e "\n${RED}Log file not found: tried $primary_log and $fallback_log${NC}"
    fi
}

# Function to monitor all logs in real-time
monitor_logs() {
    echo -e "\n${CYAN}=== REAL-TIME LOG MONITORING ===${NC}"
    echo -e "Press Ctrl+C to stop monitoring\n"
    
    # Create a temporary file to store all logs
    local temp_file=$(mktemp)
    
    # Start tail processes for each log file
    for service_config in "${SERVICES[@]}"; do
        IFS=':' read -r name display_name port primary_log fallback_log <<< "$service_config"
        local log_file
        if log_file=$(find_log_file "$primary_log" "$fallback_log"); then
            echo -e "${GREEN}Following $display_name logs from: $log_file${NC}"
            tail -f "$log_file" | sed "s/^/[$name] /" >> "$temp_file" &
        else
            echo -e "${YELLOW}Warning: No log file found for $display_name (tried: $primary_log, $fallback_log)${NC}"
        fi
    done
    
    # Monitor the combined log file
    tail -f "$temp_file" | while read line; do
        # Color code by service
        case "$line" in
            *"[nestjs]"*)
                echo -e "${BLUE}$line${NC}"
                ;;
            *"[python-ml]"*)
                echo -e "${GREEN}$line${NC}"
                ;;
            *"[python-http]"*)
                echo -e "${PURPLE}$line${NC}"
                ;;
            *"[frontend]"*)
                echo -e "${YELLOW}$line${NC}"
                ;;
            *)
                echo "$line"
                ;;
        esac
    done
    
    # Cleanup
    rm -f "$temp_file"
    pkill -f "tail -f"
}

# Function to show help
show_help() {
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  status    Show status of all services"
    echo "  logs      Show recent logs from all services"
    echo "  monitor   Monitor all logs in real-time"
    echo "  nestjs    Show only NestJS logs"
    echo "  python-ml Show only Python ML logs"
    echo "  python-http Show only Python HTTP logs"
    echo "  frontend  Show only Frontend logs"
    echo "  help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 status    # Check if all services are running"
    echo "  $0 logs      # Show recent logs from all services"
    echo "  $0 monitor   # Monitor all logs in real-time"
    echo "  $0 nestjs    # Show only NestJS logs"
}

# Main script logic
case "${1:-monitor}" in
    "status")
        show_status
        ;;
    "logs")
        show_status
        echo -e "\n${BLUE}=== RECENT LOGS ===${NC}"
        for service_config in "${SERVICES[@]}"; do
            IFS=':' read -r name display_name port primary_log fallback_log <<< "$service_config"
            case "$name" in
                "nestjs")
                    show_logs "$display_name" "$primary_log" "$fallback_log" "$BLUE"
                    ;;
                "python-ml")
                    show_logs "$display_name" "$primary_log" "$fallback_log" "$GREEN"
                    ;;
                "python-http")
                    show_logs "$display_name" "$primary_log" "$fallback_log" "$PURPLE"
                    ;;
                "frontend")
                    show_logs "$display_name" "$primary_log" "$fallback_log" "$YELLOW"
                    ;;
            esac
        done
        ;;
    "monitor")
        show_status
        monitor_logs
        ;;
    "nestjs")
        show_logs "NestJS API" ".data/logs/nestjs-api.log" "logs/nestjs-api.log" "$BLUE"
        ;;
    "python-ml")
        show_logs "Python ML" ".data/logs/python-ml.log" "logs/python-ml.log" "$GREEN"
        ;;
    "python-http")
        show_logs "Python HTTP" ".data/logs/python-http.log" "logs/python-http.log" "$PURPLE"
        ;;
    "frontend")
        show_logs "Frontend" ".data/logs/frontend.log" "logs/frontend.log" "$YELLOW"
        ;;
    "help"|"-h"|"--help")
        show_help
        ;;
    *)
        echo -e "${RED}Unknown option: $1${NC}"
        show_help
        exit 1
        ;;
esac

