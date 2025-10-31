#!/bin/bash

# Video Translator - Electron App Launcher
# This script starts only the Electron desktop app (assumes services are already running)

set -e

echo "🚀 Starting Video Translator Desktop App..."

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "package.json" ] || [ ! -d "electron" ]; then
    echo "❌ Please run this script from the project root directory"
    exit 1
fi

# Check if required services are running
print_status "Checking if required services are running..."

services_ok=true

# Check NestJS API
if ! curl -s http://localhost:3001/health > /dev/null 2>&1; then
    print_error "❌ NestJS API is not running (http://localhost:3001)"
    services_ok=false
fi

# Check Frontend
if ! curl -s http://localhost:3000 > /dev/null 2>&1; then
    print_error "❌ Frontend is not running (http://localhost:3000)"
    services_ok=false
fi

# Check Python ML Service
if ! curl -s http://localhost:50052/health > /dev/null 2>&1; then
    print_error "❌ Python ML Service is not running (http://localhost:50052)"
    services_ok=false
fi

if [ "$services_ok" = false ]; then
    print_error "Required services are not running. Please start them first:"
    echo ""
    print_status "Run: ./start-app.sh"
    echo ""
    print_status "Or start services individually:"
    print_status "  - Python ML: cd backend-python-ml && source venv/bin/activate && python src/main.py"
    print_status "  - Python HTTP: cd backend-python-ml && source venv/bin/activate && python src/http_server.py"
    print_status "  - NestJS API: cd backend-nestjs && npm run start:dev"
    print_status "  - Frontend: cd frontend && npm run dev"
    exit 1
fi

print_success "All required services are running!"

# Start Electron app
print_status "Starting Electron desktop app..."
print_status "The app will open in a new window."

NODE_ENV=development npx electron .

print_success "Electron app started successfully!"

