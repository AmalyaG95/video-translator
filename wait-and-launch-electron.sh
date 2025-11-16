#!/bin/bash
# Wait for Docker deployment to finish, then launch Electron app

set -e

echo "⏳ Waiting for Docker deployment to complete..."
echo "================================================"
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

MAX_WAIT=1800  # 30 minutes max
INTERVAL=10    # Check every 10 seconds
ELAPSED=0

check_services() {
    local all_ok=true
    
    # Check frontend
    if ! curl -s http://localhost:3000 > /dev/null 2>&1; then
        all_ok=false
    fi
    
    # Check API
    if ! curl -s http://localhost:3001 > /dev/null 2>&1; then
        all_ok=false
    fi
    
    # Check if containers are running
    local containers=$(docker ps --filter "name=translate" --format "{{.Names}}" 2>/dev/null | wc -l)
    if [ "$containers" -lt 2 ]; then
        all_ok=false
    fi
    
    echo $all_ok
}

# Check if deployment process is still running
check_deployment_running() {
    if pgrep -f "deploy.sh" > /dev/null 2>&1; then
        return 0
    fi
    return 1
}

echo "Monitoring deployment progress..."
echo ""

while [ $ELAPSED -lt $MAX_WAIT ]; do
    # Check if deployment is still running
    if check_deployment_running; then
        echo -ne "\r${BLUE}⏳ Deployment in progress... (${ELAPSED}s)${NC}"
        sleep $INTERVAL
        ELAPSED=$((ELAPSED + INTERVAL))
        continue
    fi
    
    # Deployment process finished, check services
    if [ "$(check_services)" = "true" ]; then
        echo ""
        echo ""
        echo -e "${GREEN}✅ Deployment complete! All services are running.${NC}"
        echo ""
        
        # Wait a bit more for services to fully stabilize
        sleep 5
        
        # Launch Electron
        echo -e "${BLUE}🚀 Launching Electron desktop app...${NC}"
        echo ""
        ./deploy-electron.sh
        exit 0
    else
        echo ""
        echo -e "${YELLOW}⚠️  Deployment finished but services not ready yet. Waiting...${NC}"
        sleep $INTERVAL
        ELAPSED=$((ELAPSED + INTERVAL))
    fi
done

echo ""
echo -e "${RED}❌ Timeout waiting for deployment to complete.${NC}"
echo ""
echo "Check deployment status manually:"
echo "  ./deploy.sh status"
echo "  ./deploy.sh logs"
echo ""
exit 1


