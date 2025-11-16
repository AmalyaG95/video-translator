#!/bin/bash
set -e

echo "🚀 Video Translation System - Docker Deployment"
echo "================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo -e "${RED}❌ Docker Compose is not installed. Please install Docker Compose first.${NC}"
    exit 1
fi

# Check Docker daemon is running
if ! docker info &> /dev/null; then
    echo -e "${RED}❌ Docker daemon is not running. Please start Docker first.${NC}"
    exit 1
fi

# Check if buildx is available before enabling BuildKit
# Docker Compose v2 requires buildx when BuildKit is enabled
if docker buildx version &> /dev/null 2>&1; then
    # Enable BuildKit only if buildx is available
    export DOCKER_BUILDKIT=1
    export COMPOSE_DOCKER_CLI_BUILD=1
    
    # Create a builder instance if it doesn't exist
    if ! docker buildx ls 2>/dev/null | grep -q "multibuilder"; then
        docker buildx create --name multibuilder --use --bootstrap 2>/dev/null || true
    else
        docker buildx use multibuilder 2>/dev/null || true
    fi
    echo -e "${GREEN}✅ Docker BuildKit enabled with buildx${NC}"
else
    # Disable BuildKit if buildx is not available (use legacy builder)
    export DOCKER_BUILDKIT=0
    export COMPOSE_DOCKER_CLI_BUILD=0
    echo -e "${YELLOW}⚠️  BuildKit disabled (buildx not available, using legacy builder)${NC}"
    echo -e "${YELLOW}💡 To enable BuildKit, install buildx:${NC}"
    echo -e "   ${YELLOW}   https://docs.docker.com/go/buildx/${NC}"
fi

echo -e "${GREEN}✅ Docker is ready${NC}"

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p uploads artifacts temp_work backend-python-ml/temp_work
chmod -R 755 uploads artifacts temp_work 2>/dev/null || true

# Parse command line arguments
ACTION=${1:-up}
BUILD_FLAGS=""
COMPOSE_CMD="docker-compose"

# Check if docker compose (v2) is available
if docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
fi

case "$ACTION" in
    build)
        echo "🔨 Building Docker images..."
        $COMPOSE_CMD build --no-cache
        echo -e "${GREEN}✅ Build complete${NC}"
        ;;
    up|start)
        echo "🚀 Starting services..."
        if [ "$2" == "--build" ]; then
            echo "🔨 Building images first..."
            $COMPOSE_CMD up -d --build
        else
            $COMPOSE_CMD up -d
        fi
        echo -e "${GREEN}✅ Services started${NC}"
        echo ""
        echo "📊 Service Status:"
        $COMPOSE_CMD ps
        echo ""
        echo -e "${YELLOW}📝 View logs with: ${COMPOSE_CMD} logs -f${NC}"
        echo -e "${YELLOW}🌐 Frontend: http://localhost:3000${NC}"
        echo -e "${YELLOW}🌐 API: http://localhost:3001${NC}"
        ;;
    down|stop)
        echo "🛑 Stopping services..."
        $COMPOSE_CMD down
        echo -e "${GREEN}✅ Services stopped${NC}"
        ;;
    restart)
        echo "🔄 Restarting services..."
        $COMPOSE_CMD restart
        echo -e "${GREEN}✅ Services restarted${NC}"
        ;;
    logs)
        SERVICE=${2:-""}
        if [ -z "$SERVICE" ]; then
            $COMPOSE_CMD logs -f
        else
            $COMPOSE_CMD logs -f "$SERVICE"
        fi
        ;;
    status|ps)
        $COMPOSE_CMD ps
        ;;
    clean)
        echo "🧹 Cleaning up Docker resources..."
        read -p "This will remove containers, volumes, and images. Continue? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            $COMPOSE_CMD down -v --rmi all
            echo -e "${GREEN}✅ Cleanup complete${NC}"
        else
            echo "Cancelled"
        fi
        ;;
    *)
        echo "Usage: $0 {build|up|down|restart|logs|status|clean}"
        echo ""
        echo "Commands:"
        echo "  build           - Build all Docker images"
        echo "  up [--build]    - Start all services (optionally rebuild)"
        echo "  down            - Stop all services"
        echo "  restart         - Restart all services"
        echo "  logs [service]  - View logs (optionally for specific service)"
        echo "  status          - Show service status"
        echo "  clean           - Remove containers, volumes, and images"
        echo ""
        echo "Examples:"
        echo "  $0 up              # Start services"
        echo "  $0 up --build      # Rebuild and start"
        echo "  $0 logs frontend   # View frontend logs"
        echo "  $0 logs python-ml  # View ML service logs"
        exit 1
        ;;
esac

