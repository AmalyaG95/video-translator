#!/bin/bash

# Video Translator - Enhanced Launcher with Detailed Logging
# This script starts all services and launches the Electron app with real-time monitoring

set -e

echo "🚀 Starting Video Translator..."

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_debug() {
    echo -e "${CYAN}[DEBUG]${NC} $1"
}

# Function to check service health
check_service_health() {
    local service_name=$1
    local url=$2
    local max_attempts=30
    local attempt=1
    
    print_debug "Checking $service_name health at $url..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s "$url" > /dev/null 2>&1; then
            print_success "$service_name is healthy (attempt $attempt/$max_attempts)"
            return 0
        fi
        print_debug "Waiting for $service_name... (attempt $attempt/$max_attempts)"
        sleep 2
        ((attempt++))
    done
    
    print_error "$service_name failed to become healthy after $max_attempts attempts"
    return 1
}

# Function to show service logs
show_service_logs() {
    local service_name=$1
    local log_file=$2
    local lines=${3:-10}
    
    if [ -f "$log_file" ]; then
        print_debug "Last $lines lines of $service_name logs:"
        echo "----------------------------------------"
        tail -n $lines "$log_file"
        echo "----------------------------------------"
    else
        print_warning "Log file $log_file not found for $service_name"
    fi
}

# Function to monitor service status
monitor_services() {
    print_status "Monitoring service status..."
    echo ""
    
    # Check Python ML Service
    if pgrep -f "python.*main.py" > /dev/null; then
        print_success "✅ Python ML Service is running (PID: $(pgrep -f "python.*main.py"))"
    else
        print_error "❌ Python ML Service is not running"
        show_service_logs "Python ML" ".data/logs/python-ml.log" 5
    fi
    
    # Check Python HTTP Server
    if pgrep -f "python.*http_server" > /dev/null; then
        print_success "✅ Python HTTP Server is running (PID: $(pgrep -f "python.*http_server"))"
    else
        print_error "❌ Python HTTP Server is not running"
        show_service_logs "Python HTTP" ".data/logs/python-http.log" 5
    fi
    
    # Check NestJS API
    if pgrep -f "node.*nest" > /dev/null; then
        print_success "✅ NestJS API is running (PID: $(pgrep -f "node.*nest"))"
    else
        print_error "❌ NestJS API is not running"
        show_service_logs "NestJS API" ".data/logs/nestjs-api.log" 5
    fi
    
    # Check Frontend
    if pgrep -f "next.*dev" > /dev/null; then
        print_success "✅ Frontend is running (PID: $(pgrep -f "next.*dev"))"
    else
        print_error "❌ Frontend is not running"
        show_service_logs "Frontend" ".data/logs/frontend.log" 5
    fi
    
    echo ""
}

# Function to check Electron status
check_electron_status() {
    if pgrep -f "electron.*\." > /dev/null; then
        print_success "✅ Electron App is running (PID: $(pgrep -f "electron.*\."))"
    else
        print_error "❌ Electron App is not running"
        show_service_logs "Electron" ".data/logs/electron.log" 5
    fi
}

# Check if we're in the right directory
if [ ! -f "package.json" ] || [ ! -d "frontend" ] || [ ! -d "backend-nestjs" ] || [ ! -d "backend-python-ml" ]; then
    echo "❌ Please run this script from the project root directory"
    exit 1
fi

# Ensure .data directory structure exists
print_debug "Ensuring .data directory structure exists..."
mkdir -p .data/artifacts .data/temp_work .data/uploads .data/logs

# Kill any existing processes
print_status "Cleaning up existing processes..."
pkill -f "python.*main.py" 2>/dev/null || true
pkill -f "node.*nest" 2>/dev/null || true
pkill -f "next.*dev" 2>/dev/null || true
pkill -f "python.*http_server" 2>/dev/null || true
pkill -f "electron.*\." 2>/dev/null || true
sleep 2

# Start Python ML Service
print_status "Starting Python ML Service..."
cd backend-python-ml
if [ ! -d "venv" ]; then
    print_error "Python virtual environment not found. Please run setup first."
    exit 1
fi

# Ensure .data/logs directory exists
mkdir -p ../.data/logs

source venv/bin/activate
print_debug "Activated Python virtual environment"
print_debug "Starting Python ML service with gRPC on port 50051..."

nohup python src/main.py > ../.data/logs/python-ml.log 2>&1 &
PYTHON_PID=$!
print_debug "Python ML Service started with PID: $PYTHON_PID"
cd ..

# Start Python HTTP Server for language detection
print_status "Starting Python HTTP Server..."
cd backend-python-ml
print_debug "Starting Python HTTP server on port 50052..."
nohup bash -c "source venv/bin/activate && python src/http_server.py" > ../.data/logs/python-http.log 2>&1 &
HTTP_PID=$!
print_debug "Python HTTP Server started with PID: $HTTP_PID"
cd ..

# Start NestJS API
print_status "Starting NestJS API..."
cd backend-nestjs
print_debug "Installing dependencies if needed..."
npm install --silent > /dev/null 2>&1 || true
print_debug "Starting NestJS API on port 3001..."
nohup npm run start:dev > ../.data/logs/nestjs-api.log 2>&1 &
NESTJS_PID=$!
print_debug "NestJS API started with PID: $NESTJS_PID"
cd ..

# Start Frontend
print_status "Starting Frontend..."
cd frontend
print_debug "Installing dependencies if needed..."
npm install --silent > /dev/null 2>&1 || true
print_debug "Starting Next.js frontend on port 3000..."
nohup npm run dev > ../.data/logs/frontend.log 2>&1 &
FRONTEND_PID=$!
print_debug "Frontend started with PID: $FRONTEND_PID"
cd ..

# Wait for services to start (increased time for slower systems)
print_status "Waiting for services to start..."
sleep 10

# Monitor service startup
print_status "Checking service startup status..."
monitor_services

# Health check with detailed error reporting
print_status "Performing health checks..."

# Check Python HTTP Server health (note: Python ML is gRPC on 50051, HTTP Server is on 50052)
if ! check_service_health "Python HTTP Server" "http://localhost:50052/health" 15; then
    print_error "Python HTTP Server health check failed"
    show_service_logs "Python HTTP" ".data/logs/python-http.log" 10
    print_error "Please check the logs above for errors"
fi

# Check NestJS API health
if ! check_service_health "NestJS API" "http://localhost:3001/health" 15; then
    print_error "NestJS API health check failed"
    show_service_logs "NestJS API" ".data/logs/nestjs-api.log" 10
    print_error "Please check the logs above for errors"
fi

# Check Frontend health
if ! check_service_health "Frontend" "http://localhost:3000" 15; then
    print_error "Frontend health check failed"
    show_service_logs "Frontend" ".data/logs/frontend.log" 10
    print_error "Please check the logs above for errors"
fi

# Final status check
print_status "Final service status check..."
monitor_services

# Check if all services are healthy before starting Electron
print_status "Verifying all services are ready..."

# Count healthy services
healthy_services=0
total_services=3

if curl -s "http://localhost:50052/health" > /dev/null 2>&1; then
    ((healthy_services++))
fi

if curl -s "http://localhost:3001/health" > /dev/null 2>&1; then
    ((healthy_services++))
fi

if curl -s "http://localhost:3000" > /dev/null 2>&1; then
    ((healthy_services++))
fi

print_debug "Health check results: $healthy_services/$total_services services healthy"

if [ $healthy_services -eq $total_services ]; then
    print_success "All services are healthy! ($healthy_services/$total_services)"
    
# Start Electron app
print_status "Starting Electron app..."
print_debug "Electron will open the desktop application"

# Function to show real-time status
show_realtime_status() {
    while true; do
        clear
        echo "🚀 Video Translator - Real-time Monitor"
        echo "=============================================="
        echo ""
        
        # Service status
        print_status "📊 Service Status:"
        echo "------------------"
        
if pgrep -f "python.*main.py" > /dev/null; then
            print_success "✅ Python ML Service (PID: $(pgrep -f "python.*main.py"))"
        else
            print_error "❌ Python ML Service - NOT RUNNING"
        fi
        
        if pgrep -f "python.*http_server" > /dev/null; then
            print_success "✅ Python HTTP Server (PID: $(pgrep -f "python.*http_server"))"
        else
            print_error "❌ Python HTTP Server - NOT RUNNING"
fi

if pgrep -f "node.*nest" > /dev/null; then
            print_success "✅ NestJS API (PID: $(pgrep -f "node.*nest"))"
else
            print_error "❌ NestJS API - NOT RUNNING"
fi

if pgrep -f "next.*dev" > /dev/null; then
            print_success "✅ Frontend (PID: $(pgrep -f "next.*dev"))"
        else
            print_error "❌ Frontend - NOT RUNNING"
        fi
        
        if pgrep -f "electron.*\." > /dev/null; then
            print_success "✅ Electron App (PID: $(pgrep -f "electron.*\."))"
        else
            print_error "❌ Electron App - NOT RUNNING"
        fi
        
        echo ""
        
        # Health checks
        print_status "🌐 Endpoint Health:"
        echo "-------------------"
        
        if curl -s "http://localhost:50052/health" > /dev/null 2>&1; then
            print_success "✅ Python HTTP: http://localhost:50052/health"
        else
            print_error "❌ Python HTTP: http://localhost:50052/health - NOT RESPONDING"
        fi
        
        if curl -s "http://localhost:3001/health" > /dev/null 2>&1; then
            print_success "✅ NestJS API: http://localhost:3001/health"
        else
            print_error "❌ NestJS API: http://localhost:3001/health - NOT RESPONDING"
        fi
        
        if curl -s "http://localhost:3000" > /dev/null 2>&1; then
            print_success "✅ Frontend: http://localhost:3000"
        else
            print_error "❌ Frontend: http://localhost:3000 - NOT RESPONDING"
        fi
        
        echo ""
        
        # Recent logs
        print_status "📋 Recent Logs (last 3 lines each):"
        echo "----------------------------------------"
        
        if [ -f ".data/logs/python-ml.log" ]; then
            print_debug "Python ML:"
            tail -n 3 ".data/logs/python-ml.log" | sed 's/^/  /'
        fi
        
        if [ -f ".data/logs/nestjs-api.log" ]; then
            print_debug "NestJS API:"
            tail -n 3 ".data/logs/nestjs-api.log" | sed 's/^/  /'
        fi
        
        if [ -f ".data/logs/frontend.log" ]; then
            print_debug "Frontend:"
            tail -n 3 ".data/logs/frontend.log" | sed 's/^/  /'
        fi
        
        if [ -f ".data/logs/electron.log" ]; then
            print_debug "Electron:"
            tail -n 3 ".data/logs/electron.log" | sed 's/^/  /'
        fi
        
        echo ""
        print_status "Press Ctrl+C to stop monitoring and exit"
        print_status "App is running at: http://localhost:3000"
        echo ""
        
        sleep 5
    done
}

# Trap Ctrl+C to cleanup
trap 'print_status "Shutting down services..."; pkill -f "python.*main.py" 2>/dev/null || true; pkill -f "node.*nest" 2>/dev/null || true; pkill -f "next.*dev" 2>/dev/null || true; pkill -f "python.*http_server" 2>/dev/null || true; pkill -f "electron.*\." 2>/dev/null || true; exit 0' INT

# Automatically choose monitoring mode (no monitoring to start Electron directly)
print_status "Starting Electron app directly..."
monitor_choice=3

case $monitor_choice in
    1)
        print_status "Starting real-time dashboard..."
        show_realtime_status &
        MONITOR_PID=$!
        ;;
    2)
        print_status "Starting live log monitoring..."
        print_debug "You'll see live logs from all services below:"
        echo ""
        echo "=========================================="
        echo "LIVE LOGS - Press Ctrl+C to stop all services"
        echo "=========================================="
        echo ""
        
        # Start tailing all logs
        if [ -f ".data/logs/python-ml.log" ]; then
            print_debug "Following Python ML logs..."
            tail -f .data/logs/python-ml.log | sed 's/^/[PYTHON-ML] /' &
        fi
        
        if [ -f ".data/logs/nestjs-api.log" ]; then
            print_debug "Following NestJS API logs..."
            tail -f .data/logs/nestjs-api.log | sed 's/^/[NESTJS-API] /' &
        fi
        
        if [ -f ".data/logs/frontend.log" ]; then
            print_debug "Following Frontend logs..."
            tail -f .data/logs/frontend.log | sed 's/^/[FRONTEND] /' &
        fi
        
        if [ -f ".data/logs/python-http.log" ]; then
            print_debug "Following Python HTTP logs..."
            tail -f .data/logs/python-http.log | sed 's/^/[PYTHON-HTTP] /' &
        fi
        
        if [ -f ".data/logs/electron.log" ]; then
            print_debug "Following Electron logs..."
            tail -f .data/logs/electron.log | sed 's/^/[ELECTRON] /' &
        fi
        
        # Wait a moment for logs to start
        sleep 2
        ;;
    3)
        print_status "Starting without monitoring..."
        ;;
esac

# Start Electron app
print_status "Starting Electron app..."
print_debug "Electron will open the desktop application"
nohup NODE_ENV=development npx electron . > .data/logs/electron.log 2>&1 &
ELECTRON_PID=$!
print_debug "Electron app started with PID: $ELECTRON_PID"

# Wait a moment for Electron to start
sleep 3

# Check Electron status
print_status "Checking Electron app status..."
check_electron_status
    
else
    print_error "Only $healthy_services/$total_services services are healthy. Cannot start Electron app."
    print_status "Troubleshooting information:"
    echo ""
    monitor_services
    echo ""
    print_status "Please check the logs above and fix any issues before restarting."
    print_status "You can also run individual services manually:"
    print_status "  - Python ML: cd backend-python-ml && source venv/bin/activate && python src/main.py"
    print_status "  - NestJS API: cd backend-nestjs && npm run start:dev"
    print_status "  - Frontend: cd frontend && npm run dev"
    exit 1
fi

print_success "Video Translator is now running!"
print_status "Services:"
print_status "  - Electron Desktop App: Running"
print_status "  - Frontend: http://localhost:3000"
print_status "  - NestJS API: http://localhost:3001"
print_status "  - Python ML: localhost:50051"
print_status "  - Python HTTP: localhost:50052"
print_status ""
print_status "Press Ctrl+C to stop all services"