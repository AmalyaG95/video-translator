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
SERVICES=(
    "nestjs:NestJS API:3001:logs/nestjs-api.log"
    "python-ml:Python ML:50051:logs/python-ml.log"
    "python-http:Python HTTP:50052:logs/python-http.log"
    "frontend:Frontend:3000:logs/frontend.log"
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
        IFS=':' read -r name display_name port log_file <<< "$service_config"
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

# Function to display logs for a specific service
show_logs() {
    local service_name=$1
    local log_file=$2
    local color=$3
    
    if [ -f "$log_file" ]; then
        echo -e "\n${color}=== $service_name LOGS (last 20 lines) ===${NC}"
        tail -20 "$log_file" | sed "s/^/${color}[$service_name]${NC} /"
    else
        echo -e "\n${RED}Log file not found: $log_file${NC}"
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
        IFS=':' read -r name display_name port log_file <<< "$service_config"
        if [ -f "$log_file" ]; then
            tail -f "$log_file" | sed "s/^/[$name] /" >> "$temp_file" &
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
            IFS=':' read -r name display_name port log_file <<< "$service_config"
            case "$name" in
                "nestjs")
                    show_logs "$display_name" "$log_file" "$BLUE"
                    ;;
                "python-ml")
                    show_logs "$display_name" "$log_file" "$GREEN"
                    ;;
                "python-http")
                    show_logs "$display_name" "$log_file" "$PURPLE"
                    ;;
                "frontend")
                    show_logs "$display_name" "$log_file" "$YELLOW"
                    ;;
            esac
        done
        ;;
    "monitor")
        show_status
        monitor_logs
        ;;
    "nestjs")
        show_logs "NestJS API" "logs/nestjs-api.log" "$BLUE"
        ;;
    "python-ml")
        show_logs "Python ML" "logs/python-ml.log" "$GREEN"
        ;;
    "python-http")
        show_logs "Python HTTP" "logs/python-http.log" "$PURPLE"
        ;;
    "frontend")
        show_logs "Frontend" "logs/frontend.log" "$YELLOW"
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

