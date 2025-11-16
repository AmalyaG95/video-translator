#!/bin/bash
# Desktop App Deployment Script
# Connects Electron app to Docker services

set -e

echo "🖥️  Video Translator - Desktop App Deployment"
echo "==============================================="
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if Docker services are running
print_status "Checking Docker services..."

services_ok=true

# Check Frontend
if ! curl -s http://localhost:3000 > /dev/null 2>&1; then
    print_error "❌ Frontend is not running (http://localhost:3000)"
    services_ok=false
else
    print_success "✅ Frontend is running"
fi

# Check NestJS API
if ! curl -s http://localhost:3001/health > /dev/null 2>&1 && ! curl -s http://localhost:3001 > /dev/null 2>&1; then
    print_error "❌ NestJS API is not running (http://localhost:3001)"
    services_ok=false
else
    print_success "✅ NestJS API is running"
fi

# Check Python ML Service (gRPC - harder to check, but we can try)
if ! docker ps --format "{{.Names}}" | grep -q "python-ml"; then
    print_warning "⚠️  Python ML service container not found"
else
    print_success "✅ Python ML service container is running"
fi

if [ "$services_ok" = false ]; then
    print_error "Required services are not running."
    echo ""
    print_status "Please start Docker services first:"
    echo ""
    echo "  ./deploy.sh up"
    echo ""
    print_status "Or check service status:"
    echo ""
    echo "  ./deploy.sh status"
    echo ""
    exit 1
fi

print_success "All services are running!"

# Check if Electron is installed
print_status "Checking Electron dependencies..."

if [ ! -d "node_modules/electron" ]; then
    print_status "Installing Electron dependencies..."
    npm install
fi

print_success "Dependencies ready!"

# Start Electron app
print_status "Starting Electron desktop app..."
print_status "The app will connect to Docker services at:"
print_status "  - Frontend: http://localhost:3000"
print_status "  - API: http://localhost:3001"
print_status "  - ML Service: localhost:50051 (gRPC)"
echo ""

# Set environment for Docker mode
export NODE_ENV=development
export USE_DOCKER=true

print_status "Launching Electron app..."
echo ""

# Start Electron
npx electron .

print_success "Desktop app launched successfully!"


