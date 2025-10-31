#!/bin/bash
# Run full production-like mode with all services in Docker
# Usage: ./run-dev-full-docker.sh

set -e

cd "$(dirname "$0")"

echo "🐳 Starting Full Docker Mode (Production-like)"
echo "=============================================="
echo ""

# Stop and remove existing containers
echo "🛑 Stopping existing containers..."
docker-compose down

# Build and start all services
echo "🔨 Building Docker images..."
docker-compose build

echo "🚀 Starting all services..."
docker-compose up -d

echo ""
echo "⏳ Waiting for services to be ready..."
sleep 5

echo ""
echo "✅ All services started!"
echo ""
echo "Services:"
echo "  🐍 Python ML:  localhost:50051 (gRPC)"
echo "  🟢 NestJS API: localhost:3001"
echo "  ⚛️  Frontend:   localhost:3000"
echo ""
echo "📱 Access the application:"
echo "   http://localhost:3000"
echo ""
echo "📋 View logs:"
echo "   docker-compose logs -f"
echo ""
echo "🛑 Stop all services:"
echo "   docker-compose down"
echo ""


